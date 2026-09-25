"""
Audit log — one table, one write per proxied call.

Raw asyncpg, not an ORM: this is a single append-only table with exactly
one query shape (INSERT), no joins, no reads on the hot path, and no
migrations beyond the CREATE TABLE below. SQLAlchemy would add a mapping
layer that earns nothing here — reach for it if this grows a second table
or any read path beyond ops/debugging queries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import asyncpg

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    pii_found BOOLEAN NOT NULL,
    topic_blocked BOOLEAN NOT NULL,
    action TEXT NOT NULL,
    latency_ms INTEGER NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO audit_log (provider, request_hash, pii_found, topic_blocked, action, latency_ms)
VALUES ($1, $2, $3, $4, $5, $6);
"""


@dataclass
class AuditEntry:
    provider: str
    request_hash: str
    pii_found: bool
    topic_blocked: bool
    action: str
    latency_ms: int


def hash_request(prompt: str) -> str:
    """
    Hash, never store, the raw prompt. The audit log is a compliance trail
    (did we redact, did we block, how long did it take), not a second copy
    of everything a user typed — storing the plaintext prompt here would
    quietly defeat the PII redaction this whole service exists to do.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "AuditLog":
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def record(self, entry: AuditEntry) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                INSERT_SQL,
                entry.provider,
                entry.request_hash,
                entry.pii_found,
                entry.topic_blocked,
                entry.action,
                entry.latency_ms,
            )

    async def ping(self) -> bool:
        """Used by /health — confirms the pool can actually reach Postgres,
        not just that it was constructed successfully at startup."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False
