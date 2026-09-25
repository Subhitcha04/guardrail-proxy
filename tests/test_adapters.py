import httpx
import pytest

from app.adapters.groq_adapter import GroqAdapter
from app.adapters.openai_adapter import OpenAIAdapter

ADAPTERS = [
    (GroqAdapter, "GROQ_API_KEY"),
    (OpenAIAdapter, "OPENAI_API_KEY"),
]


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("adapter_cls,env_var", ADAPTERS)
async def test_complete_returns_message_content(adapter_cls, env_var):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == adapter_cls.BASE_URL
        body = request.read()
        assert b'"model"' in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]})

    client = make_client(handler)
    adapter = adapter_cls(client=client, api_key="test-key", model="a-model")
    try:
        assert await adapter.complete("hello") == "hi there"
    finally:
        await client.aclose()


@pytest.mark.parametrize("adapter_cls,env_var", ADAPTERS)
async def test_complete_retries_once_then_raises_on_repeated_5xx(adapter_cls, env_var):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(500)

    client = make_client(handler)
    adapter = adapter_cls(client=client, api_key="test-key")
    try:
        with pytest.raises(RuntimeError):
            await adapter.complete("hello")
    finally:
        await client.aclose()

    # Exactly the documented retry budget -- one retry, not a backoff loop.
    assert call_count["n"] == 2


@pytest.mark.parametrize("adapter_cls,env_var", ADAPTERS)
async def test_complete_raises_clearly_when_api_key_missing(adapter_cls, env_var, monkeypatch):
    monkeypatch.delenv(env_var, raising=False)
    client = make_client(lambda request: httpx.Response(200, json={}))
    adapter = adapter_cls(client=client, api_key=None)
    try:
        with pytest.raises(RuntimeError, match=env_var):
            await adapter.complete("hello")
    finally:
        await client.aclose()


@pytest.mark.parametrize("adapter_cls,env_var", ADAPTERS)
async def test_error_message_includes_actual_response_body(adapter_cls, env_var):
    # Regression test for a real bug: the original code swallowed the
    # response body on a 4xx/5xx, so a decommissioned-model error and a
    # generic outage looked identical from the caller's side. The body is
    # where the provider actually says *why* -- surface it.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": {"message": "The model has been decommissioned", "code": "model_decommissioned"}},
        )

    client = make_client(handler)
    adapter = adapter_cls(client=client, api_key="test-key")
    try:
        with pytest.raises(RuntimeError, match="model_decommissioned"):
            await adapter.complete("hello")
    finally:
        await client.aclose()


@pytest.mark.parametrize("adapter_cls,env_var", ADAPTERS)
async def test_malformed_response_is_treated_as_a_failure(adapter_cls, env_var):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = make_client(handler)
    adapter = adapter_cls(client=client, api_key="test-key")
    try:
        with pytest.raises(RuntimeError):
            await adapter.complete("hello")
    finally:
        await client.aclose()
