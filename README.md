# Guardrail Proxy

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)

An async LLM gateway that enforces PII redaction, topic-denial policy
screening, and structured audit logging in front of Groq and OpenAI chat
completions — a provider-agnostic guardrail layer for any application
integrating third-party LLM APIs.

Verified end-to-end against live infrastructure: real Groq completions,
real PostgreSQL writes, real Presidio/spaCy PII detection. 33/33 automated
tests passing, including a live-database integration test.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Extending: Adding a Provider](#extending-adding-a-provider)
- [Known Limitations & Design Decisions](#known-limitations--design-decisions)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

Guardrail Proxy sits between a client and an upstream LLM provider,
enforcing three policies on every request before a response is returned:

1. **Topic denial** — the inbound prompt is screened against a configurable
   set of disallowed topics before any provider call is made; the
   provider's response is screened again after generation, as defense in
   depth against an innocuous prompt producing disallowed output.
2. **PII redaction** — provider output is scanned for personally
   identifiable information (names, emails, phone numbers) and redacted
   before being returned to the client.
3. **Audit logging** — every request is logged to PostgreSQL with an
   identical schema regardless of which provider served it, capturing
   policy outcomes and latency without ever storing the raw prompt text.

The proxy exposes an OpenAI-compatible `/v1/chat/completions` endpoint,
making it a drop-in addition to existing integrations.

## Architecture

```mermaid
flowchart LR
    A[Client Request] --> B{Topic Pre-Check<br/>on Prompt}
    B -->|Blocked| C[422 Blocked Response]
    B -->|Clear| D[Provider Adapter<br/>Groq / OpenAI]
    D --> E[Raw Completion]
    E --> F[PII Check<br/>Presidio]
    E --> G[Topic Check<br/>on Output]
    F --> H{Apply Policy}
    G --> H
    H -->|Blocked| C
    H -->|Redacted / Allowed| I[Response to Client]
    D -.-> J[(PostgreSQL<br/>Audit Log)]
    H -.-> J
```

Topic denial runs twice by design: once cheaply on the prompt (skipping the
provider call entirely for obvious cases) and once on the output (catching
cases where an innocuous prompt still produces disallowed content).

## Features

- **Provider-agnostic core** — adapters implement a single `Protocol`
  (`async def complete(prompt: str) -> str`); adding a new provider
  requires one new file and one registry entry, with zero changes to
  policy, proxy, or audit logic.
- **PII detection and redaction** via Microsoft Presidio, restricted to an
  explicit, empirically validated entity allowlist rather than Presidio's
  noisy default configuration (see [Known Limitations](#known-limitations--design-decisions)).
- **Two-stage topic-denial screening** using sentence-embedding cosine
  similarity (`sentence-transformers`), configurable per-topic via
  `policy.yaml`.
- **Structured, schema-consistent audit logging** to PostgreSQL — prompts
  are hashed (SHA-256), never stored in plaintext.
- **Fully asynchronous** — FastAPI, `httpx.AsyncClient`, `asyncpg`
  throughout; synchronous CPU-bound work (Presidio, embedding inference) is
  offloaded via `asyncio.to_thread` to keep the event loop unblocked.
- **Configuration-driven policy** — redaction and blocking behavior defined
  declaratively in `policy.yaml`, validated at startup via Pydantic.
- **33 automated tests** spanning unit, integration, and live-database
  coverage.

## Tech Stack

| Layer            | Technology                                  |
|-------------------|----------------------------------------------|
| API framework     | FastAPI + Uvicorn                            |
| HTTP client       | httpx (async)                                |
| PII detection      | Microsoft Presidio + spaCy (`en_core_web_lg`) |
| Semantic screening | sentence-transformers (`all-MiniLM-L6-v2`)   |
| Data validation    | Pydantic v2                                  |
| Database           | PostgreSQL via asyncpg                       |
| Testing            | pytest, pytest-asyncio, httpx `MockTransport` |
| Containerization   | Docker, docker-compose                       |

## Prerequisites

- Python 3.12+
- PostgreSQL 14+ (or Docker, via the provided `docker-compose.yml`)
- A Groq and/or OpenAI API key

## Installation

```bash
git clone https://github.com/Subhitcha04/guardrail-proxy.git
cd guardrail-proxy

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

`sentence-transformers` pulls in `torch` (several hundred MB) and downloads
the `all-MiniLM-L6-v2` embedding model (~80MB) from Hugging Face on first
run.

## Configuration

```bash
cp .env.example .env
```

| Variable                | Description                                      | Default                                                    |
|--------------------------|---------------------------------------------------|--------------------------------------------------------------|
| `GROQ_API_KEY`           | Groq API key                                       | *(required for Groq requests)*                                |
| `OPENAI_API_KEY`         | OpenAI API key                                     | *(required for OpenAI requests)*                               |
| `DATABASE_DSN`           | PostgreSQL connection string for the audit log    | `postgresql://postgres:postgres@localhost:5432/guardrail`     |
| `POLICY_PATH`            | Path to the policy configuration file             | `policy.yaml`                                                  |
| `EMBEDDING_MODEL_NAME`   | sentence-transformers model for topic screening   | `all-MiniLM-L6-v2`                                             |

**Operational note:** Groq's model catalog changes without much notice —
`llama-3.1-8b-instant` was decommissioned for free/developer-tier accounts
on 2026-08-16. This project defaults to `openai/gpt-oss-20b` (a Groq-hosted
model, despite the prefix). Check
[console.groq.com/docs/models](https://console.groq.com/docs/models) if
requests start returning 404s.

## Running the Service

**Local:**

```bash
export $(cat .env | xargs)
uvicorn app.main:app --reload
```

**Docker Compose** (application + PostgreSQL):

```bash
docker compose up --build
```

The service listens on `http://localhost:8000`.

## API Reference

| Method | Endpoint                | Description                                      |
|--------|---------------------------|----------------------------------------------------|
| `POST` | `/v1/chat/completions`   | Proxies a chat completion through the guardrail pipeline |
| `GET`  | `/health`                | Reports database connectivity and model load status |

### `POST /v1/chat/completions`

Request model is namespaced `<provider>/<model-name>`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

```json
{
  "model": "groq/openai/gpt-oss-20b",
  "blocked": false,
  "block_reason": null,
  "pii_found": false,
  "content": "The capital of France is Paris."
}
```

A request that trips the topic-denial policy is rejected with HTTP 422
before any provider call is made:

```json
{"blocked": true, "reason": "topic_denial:medical_advice"}
```

### `GET /health`

```json
{"status": "ok", "database": true, "embedding_model_loaded": true}
```

## Testing

```bash
pytest -v
```

25 tests run against nothing but the local codebase. One test
(`test_record_writes_a_row_with_identical_schema_across_providers`)
requires a live PostgreSQL instance:

```bash
TEST_DATABASE_DSN=postgresql://postgres:postgres@localhost:5432/guardrail pytest -v
```

`test_service.py` requires `sentence-transformers` to be importable, the
same requirement as running the application itself.

## Project Structure

```
guardrail-proxy/
├── app/
│   ├── main.py                # FastAPI app, lifespan startup, routing
│   ├── service.py             # GuardrailProxy -- the composed request flow
│   ├── policy.py               # policy.yaml -> validated Pydantic model
│   ├── audit.py                # PostgreSQL audit log
│   ├── adapters/
│   │   ├── base.py             # LLMProvider Protocol
│   │   ├── groq_adapter.py
│   │   └── openai_adapter.py
│   └── checks/
│       ├── pii.py               # Presidio-based PII detection/redaction
│       └── topic_denial.py      # Embedding-based topic screening
├── tests/
├── policy.yaml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Extending: Adding a Provider

1. Create `app/adapters/your_adapter.py` implementing:
   ```python
   async def complete(self, prompt: str) -> str: ...
   ```
2. Register it in `app/main.py`:
   ```python
   PROVIDER_REGISTRY["yourprovider"] = YourAdapter
   ```

No changes to `policy.py`, `service.py`, or `audit.py` are required —
enforced by `test_audit_log_schema_identical_across_providers` and
`test_topic_block_fires_identically_regardless_of_provider`.

## Known Limitations & Design Decisions

**PII detection scope.** Redaction uses an explicit allowlist
(`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`) rather than Presidio's full
default recognizer set. This followed empirical testing against live model
output, which surfaced three false-positive categories in Presidio's stock
configuration:
- `LOCATION` and `DATE_TIME` — NLP-driven recognizers that flagged ordinary
  words ("France," "quarterly") with no privacy relevance.
- `IN_PAN` — a locale-specific recognizer (India tax ID), active by default
  regardless of the input's actual locale, which flagged plain English
  words in a support-ticket template.

Additional entity types (`CREDIT_CARD`, `US_SSN`, `IBAN_CODE`, etc.) are
reasonable extensions but are intentionally excluded until validated
against representative output.

**PERSON detection ceiling.** Presidio's `SpacyRecognizer` assigns a
constant confidence score (0.85) to every `PERSON` match, so
confidence-based filtering cannot distinguish a genuine name from an NER
false positive (e.g., two adjacent capitalized words incorrectly grouped
as a name span). No threshold-based mitigation is possible without a
custom recognizer or an alternative NER model; this is documented as an
open problem rather than addressed with an unvalidated heuristic.

**Topic reference set.** Each denied topic is represented by a single
reference sentence (`DEFAULT_TOPIC_REFERENCES` in `topic_denial.py`).
Sufficient for direct requests; a paraphrased or obfuscated prompt will
score lower against a single reference than it should. Production use
should expand this to multiple paraphrases per topic with averaged
embeddings.

**Dependency pinning.** `numpy` is pinned to `1.26.4`. Without this pin,
dependency resolution can select `numpy>=2.0`, which is binary-incompatible
with `thinc==8.2.5`'s compiled extensions and raises an import-time ABI
error.

## Roadmap

- Per-entity-type confidence thresholds, configurable via `policy.yaml`
- Multi-paraphrase topic-reference embeddings with score averaging
- Rate limiting and per-client quota enforcement
- Structured request tracing / OpenTelemetry integration
- Streaming response support
