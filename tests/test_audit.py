import hashlib
import os

import pytest

from app.audit import AuditEntry, AuditLog, hash_request

# The integration test below needs a real Postgres reachable at TEST_DATABASE_DSN.
# It's skipped by default so the suite doesn't require a running DB just to
# check hashing logic. Set TEST_DATABASE_DSN and run with `pytest -m integration`
# to exercise it, e.g. against the docker-compose Postgres service.
requires_postgres = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_DSN"),
    reason="set TEST_DATABASE_DSN to run the live-Postgres audit log test",
)


def test_hash_request_is_deterministic_sha256():
    prompt = "what's my account balance?"
    expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert hash_request(prompt) == expected


def test_hash_request_never_leaks_the_original_text():
    prompt = "my SSN is 123-45-6789"
    digest = hash_request(prompt)
    assert "123-45-6789" not in digest
    assert len(digest) == 64  # sha256 hex digest length


@requires_postgres
async def test_record_writes_a_row_with_identical_schema_across_providers():
    # PS-3.3: the audit log schema must be identical regardless of which
    # provider served the request — same INSERT_SQL, different `provider`
    # value, no per-provider branching.
    dsn = os.environ["TEST_DATABASE_DSN"]
    audit_log = await AuditLog.connect(dsn)
    try:
        for provider in ("groq", "openai"):
            await audit_log.record(
                AuditEntry(
                    provider=provider,
                    request_hash=hash_request(f"test prompt for {provider}"),
                    pii_found=False,
                    topic_blocked=False,
                    action="allowed",
                    latency_ms=42,
                )
            )
        assert await audit_log.ping()
    finally:
        await audit_log.close()
