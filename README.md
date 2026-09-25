# guardrail-proxy

An async LLM gateway that sits in front of Groq and OpenAI: every completion
gets PII-redacted and topic-screened before it comes back, with every call
logged to Postgres regardless of which provider served it.

```
request → adapter.complete() → raw output → PII check → topic check → policy actions → response
                                                                              ↓
                                                                         audit_log
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

`sentence-transformers` pulls in `torch` — a few hundred MB. It, and the
`all-MiniLM-L6-v2` embedding model itself (~80MB, downloaded from
huggingface.co on first run), need a normal machine with normal disk and
normal internet; this was built and code-verified in a disk-constrained
sandbox that couldn't fit torch, so that specific install path hasn't been
run end-to-end yet — see **Known gaps** below.

Copy `.env.example` to `.env` and fill in at least one provider key:

```bash
cp .env.example .env
# edit .env: GROQ_API_KEY=... and/or OPENAI_API_KEY=...
```

Postgres: either run one locally and set `DATABASE_DSN`, or just use
docker-compose (below), which brings up Postgres for you.

## Running

**Local:**
```bash
export $(cat .env | xargs)   # or use direnv / your shell's env loading
uvicorn app.main:app --reload
```

**Docker (app + Postgres together):**
```bash
docker compose up --build
```

Either way, the service comes up on `http://localhost:8000`.

## Usage

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq/llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

`model` is namespaced `<provider>/<model-name>` — `groq/llama-3.1-8b-instant`,
`openai/gpt-4o-mini`, etc. A response that trips the topic-denial policy
comes back as HTTP 422 with `{"blocked": true, "reason": "topic_denial:<topic>"}`
instead of content.

```bash
curl http://localhost:8000/health
# {"status": "ok", "database": true, "embedding_model_loaded": true}
```

## Testing

```bash
pytest -v
```

19 of the tests need nothing but this repo's own code and `pip install`.
One (`test_record_writes_a_row_with_identical_schema_across_providers`) needs
a real Postgres — set `TEST_DATABASE_DSN` to run it, e.g. against the
docker-compose `db` service on `localhost:5432`. `test_service.py` needs
`sentence-transformers` actually importable, since `service.py` imports
`topic_denial.py` at module load time.

## Adding a third provider

1. `app/adapters/your_adapter.py` — implement `async def complete(self, prompt: str) -> str`
   (see `groq_adapter.py`).
2. Register it: `PROVIDER_REGISTRY["yourprovider"] = YourAdapter` in `main.py`.

That's it — `policy.py`, `service.py`, `audit.py` don't change. This is the
thing Phase 7's test suite actually checks (`test_service.py`'s
`test_audit_log_schema_identical_across_providers` and
`test_topic_block_fires_identically_regardless_of_provider`).

## Known gaps — read before demoing

- **Topic denial screens the output, not the input** (per the build doc's
  stated flow: adapter call happens *before* PII/topic checks). A
  banned-topic prompt still costs a full provider call before getting
  blocked. Pre-screening the prompt too is a few lines in
  `GuardrailProxy.handle()`, not a redesign — worth doing if API cost
  matters more than matching the doc's exact flow.
- **Presidio's default recognizers produce false positives.** Confirmed by
  running it: the bare word "quarterly" gets flagged as `DATE_TIME` at
  score 0.85. Nothing currently filters findings by confidence — every
  finding above score 0 gets redacted. Fix is a per-entity-type minimum
  score in `policy.yaml`, not yet implemented.
- **`numpy` is pinned to `1.26.4`, on purpose.** Without the pin, pip's
  resolver can pick `numpy>=2.0` for `sentence-transformers`' generic numpy
  requirement, which is binary-incompatible with `thinc==8.2.5`'s compiled
  extensions (hit `numpy.dtype size changed, may indicate binary
  incompatibility` while building this). Don't remove the pin without
  retesting `import app.checks.pii` afterward.
- **Topic reference set is one sentence per topic** (`DEFAULT_TOPIC_REFERENCES`
  in `topic_denial.py`). Thin on purpose for a demo; a paraphrased or
  obfuscated banned-topic request will score lower against one reference
  sentence than it should. Add 3-5 paraphrases per topic and average their
  embeddings before this goes near a real obfuscation test.
- **`test_service.py` and the topic-denial/proxy-wiring paths are
  syntax-checked but not execution-tested** in the environment this was
  built in — no disk headroom for torch. Run `pytest -v` yourself after
  `pip install -r requirements.txt` on a normal machine to confirm; nothing
  about the code depends on the sandbox, but "I wrote it" and "I ran it"
  aren't the same claim and only the latter is true for these specific
  modules right now.

## Structure

```
guardrail-proxy/
  app/
    main.py              # FastAPI app + lifespan (model/DB/client startup)
    adapters/
      base.py             # LLMProvider Protocol
      groq_adapter.py
      openai_adapter.py
    checks/
      pii.py               # Presidio wrapper
      topic_denial.py      # sentence-transformers cosine-similarity screen
    policy.py              # policy.yaml -> Pydantic model
    audit.py               # Postgres audit_log writes
  tests/
  policy.yaml
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```
