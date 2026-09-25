"""
FastAPI entrypoint.

Everything expensive — the embedding model, Presidio's spaCy pipeline, the
Postgres pool, the shared httpx client — is constructed exactly once in
`lifespan()` and handed to route handlers through `Depends()`. Pitfall #3:
this is deliberately *not* one stateful class holding the DB connection,
the model, and the HTTP client as instance attributes (the Java-OOP
instinct). Each dependency is a small, independently swappable thing —
which is also what makes route handlers testable without a real Postgres
or a real embedding model: swap the dependency, not the class.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.adapters.base import LLMProvider
from app.adapters.groq_adapter import GroqAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.audit import AuditLog
from app.checks.pii import PIIChecker
from app.checks.topic_denial import TopicDenialChecker
from app.policy import Policy
from app.service import GuardrailProxy

POLICY_PATH = os.environ.get("POLICY_PATH", "policy.yaml")
DATABASE_DSN = os.environ.get(
    "DATABASE_DSN", "postgresql://postgres:postgres@localhost:5432/guardrail"
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# provider prefix -> adapter class. Adding a third provider is one new
# adapter file plus one new line here — zero changes to policy.py or
# service.py. That's the specific bar Phase 7's test harness checks.
PROVIDER_REGISTRY: dict[str, type] = {
    "groq": GroqAdapter,
    "openai": OpenAIAdapter,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    policy = Policy.from_yaml(POLICY_PATH)

    # ~80MB, loads in a second or two on CPU. Loaded exactly once, here,
    # for the life of the process — see pitfall #1 in the README.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    pii_checker = PIIChecker(analyzer=analyzer, anonymizer=anonymizer)
    topic_checker = TopicDenialChecker(
        model=embedding_model,
        topics=policy.checks.topic_denial.topics,
        threshold=policy.checks.topic_denial.threshold,
    )

    audit_log = await AuditLog.connect(DATABASE_DSN)
    http_client = httpx.AsyncClient()

    app.state.policy = policy
    app.state.pii_checker = pii_checker
    app.state.topic_checker = topic_checker
    app.state.audit_log = audit_log
    app.state.http_client = http_client
    app.state.embedding_model = embedding_model

    yield

    await audit_log.close()
    await http_client.aclose()


app = FastAPI(title="Guardrail Proxy", lifespan=lifespan)


# --- Dependencies ------------------------------------------------------
# Each pulls exactly one thing off app.state. Swap any one of these in a
# test (a fake AuditLog, a fake GuardrailProxy) without touching the others.

def get_proxy() -> GuardrailProxy:
    return GuardrailProxy(
        policy=app.state.policy,
        pii_checker=app.state.pii_checker,
        topic_checker=app.state.topic_checker,
        audit_log=app.state.audit_log,
    )


def get_http_client() -> httpx.AsyncClient:
    return app.state.http_client


def resolve_provider(model: str, client: httpx.AsyncClient) -> tuple[str, LLMProvider]:
    """
    `model` is namespaced as "<provider>/<model-name>" — e.g.
    "groq/llama-3.1-8b-instant" — mirroring how OpenRouter and similar
    multi-provider gateways disambiguate which backend a model name
    belongs to.
    """
    if "/" not in model:
        raise HTTPException(
            status_code=400,
            detail=f"model must be '<provider>/<model-name>', got '{model}'",
        )
    provider_name, model_name = model.split("/", 1)
    adapter_cls = PROVIDER_REGISTRY.get(provider_name)
    if adapter_cls is None:
        raise HTTPException(status_code=400, detail=f"unknown provider '{provider_name}'")
    return provider_name, adapter_cls(client=client, model=model_name)


# --- Request/response models --------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)


class ChatCompletionResponse(BaseModel):
    model: str
    blocked: bool
    block_reason: str | None
    pii_found: bool
    content: str


# --- Routes ---------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    proxy: GuardrailProxy = Depends(get_proxy),
    client: httpx.AsyncClient = Depends(get_http_client),
) -> ChatCompletionResponse:
    provider_name, provider = resolve_provider(body.model, client)

    # Only the last message goes through as the prompt — this mirrors
    # OpenAI's request shape for drop-in compatibility, but this proxy
    # does not do multi-turn context assembly. That's a deliberate scope
    # cut for the hackathon build, not an oversight — flag it in the demo.
    prompt = body.messages[-1].content

    result = await proxy.handle(provider_name, provider, prompt)

    if result.blocked:
        raise HTTPException(
            status_code=422, detail={"blocked": True, "reason": result.block_reason}
        )

    return ChatCompletionResponse(
        model=body.model,
        blocked=result.blocked,
        block_reason=result.block_reason,
        pii_found=result.pii_found,
        content=result.content,
    )


@app.get("/health")
async def health() -> dict:
    """Checks the two things worth checking: DB connectivity and that the
    embedding model actually loaded — not just that the process is up."""
    db_ok = await app.state.audit_log.ping()
    model_ok = getattr(app.state, "embedding_model", None) is not None
    status = "ok" if (db_ok and model_ok) else "degraded"
    return {"status": status, "database": db_ok, "embedding_model_loaded": model_ok}
