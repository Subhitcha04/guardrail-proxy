"""
Phase 5: the actual guardrail proxy.

request in -> topic pre-check on the prompt -> (if clear) adapter call ->
raw output -> PII check -> topic check -> apply policy actions ->
response out.

This stays as one composed method rather than being spread across route
handler logic (pitfall #3's cousin: a fat route function is the same
god-object problem wearing a different hat). Every problem statement in
the source doc is really testing this one call sequence, so it should be
readable top to bottom in a single place.

Topic denial runs twice, deliberately, not redundantly: once on the
inbound prompt (cheap, skips the provider call entirely for the obvious
cases) and once on the provider's raw output (defense in depth -- an
innocuous-looking prompt can still produce output that trips the policy,
and only the output check can catch that). The original build doc's
stated flow only checked the output; this was a known gap, closed here
rather than left documented.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.adapters.base import LLMProvider
from app.audit import AuditEntry, AuditLog, hash_request
from app.checks.pii import PIIChecker
from app.checks.topic_denial import TopicDenialChecker
from app.policy import CheckAction, Policy


@dataclass
class ProxyResult:
    content: str
    blocked: bool
    pii_found: bool
    topic_blocked: bool
    block_reason: str | None = None


class GuardrailProxy:
    def __init__(
        self,
        policy: Policy,
        pii_checker: PIIChecker,
        topic_checker: TopicDenialChecker,
        audit_log: AuditLog,
    ) -> None:
        self._policy = policy
        self._pii = pii_checker
        self._topic = topic_checker
        self._audit = audit_log

    async def handle(self, provider_name: str, provider: LLMProvider, prompt: str) -> ProxyResult:
        start = time.perf_counter()

        # Pre-screen the prompt itself before spending a provider call on
        # a request the policy will block anyway.
        input_topic_result = await self._topic.check(prompt)
        input_blocked = (
            input_topic_result.blocked
            and self._policy.checks.topic_denial.action == CheckAction.BLOCK
        )
        if input_blocked:
            result = ProxyResult(
                content="",
                blocked=True,
                pii_found=False,
                topic_blocked=True,
                block_reason=f"topic_denial:{input_topic_result.matched_topic}",
            )
            await self._record(provider_name, prompt, result, start)
            return result

        raw_output = await provider.complete(prompt)

        pii_result = await self._pii.check(raw_output)
        topic_result = await self._topic.check(raw_output)

        topic_blocked = (
            topic_result.blocked
            and self._policy.checks.topic_denial.action == CheckAction.BLOCK
        )

        if topic_blocked:
            final_text = ""
            blocked = True
            block_reason = f"topic_denial:{topic_result.matched_topic}"
        else:
            blocked = False
            block_reason = None
            if pii_result.found_any and self._policy.checks.pii.action == CheckAction.REDACT:
                final_text = pii_result.redacted_text
            else:
                final_text = raw_output

        result = ProxyResult(
            content=final_text,
            blocked=blocked,
            pii_found=pii_result.found_any,
            topic_blocked=topic_result.blocked,
            block_reason=block_reason,
        )
        await self._record(provider_name, prompt, result, start)
        return result

    async def _record(
        self, provider_name: str, prompt: str, result: ProxyResult, start: float
    ) -> None:
        latency_ms = int((time.perf_counter() - start) * 1000)
        action = "blocked" if result.blocked else ("redacted" if result.pii_found else "allowed")
        await self._audit.record(
            AuditEntry(
                provider=provider_name,
                request_hash=hash_request(prompt),
                pii_found=result.pii_found,
                topic_blocked=result.topic_blocked,
                action=action,
                latency_ms=latency_ms,
            )
        )
