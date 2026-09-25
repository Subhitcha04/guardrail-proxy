"""
Tests the composed flow in service.py using fakes for every collaborator
(provider, PII checker, topic checker, audit log) -- none of these need a
real embedding model, real Postgres, or a real provider API key. This is
deliberate: GuardrailProxy.handle() is the thing every problem statement
is really testing, so it should be testable in complete isolation from
the heavy dependencies the individual checks pull in.

Note: this file imports app.service, which imports app.checks.topic_denial,
which imports sentence_transformers. It needs that package installed to
even collect -- see test_pii.py / test_adapters.py for the checks that run
without it.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.audit import AuditEntry
from app.policy import CheckAction, ChecksConfig, PIICheckConfig, Policy, TopicDenialConfig
from app.service import GuardrailProxy


class FakeProvider:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._response


@dataclass
class _FakePIIResult:
    redacted_text: str
    found_any: bool


class FakePIIChecker:
    def __init__(self, found_any: bool, redacted_text: str = ""):
        self._found_any = found_any
        self._redacted_text = redacted_text

    async def check(self, text: str) -> _FakePIIResult:
        return _FakePIIResult(redacted_text=self._redacted_text or text, found_any=self._found_any)


@dataclass
class _FakeTopicResult:
    blocked: bool
    matched_topic: str | None
    score: float


class FakeTopicChecker:
    def __init__(self, blocked: bool, matched_topic: str | None = None):
        self._blocked = blocked
        self._matched_topic = matched_topic

    async def check(self, text: str) -> _FakeTopicResult:
        return _FakeTopicResult(
            blocked=self._blocked,
            matched_topic=self._matched_topic,
            score=0.9 if self._blocked else 0.1,
        )


class FakeAuditLog:
    def __init__(self):
        self.entries: list[AuditEntry] = []

    async def record(self, entry: AuditEntry) -> None:
        self.entries.append(entry)


def make_policy(
    pii_action: CheckAction = CheckAction.REDACT,
    topic_action: CheckAction = CheckAction.BLOCK,
) -> Policy:
    return Policy(
        checks=ChecksConfig(
            pii=PIICheckConfig(action=pii_action),
            topic_denial=TopicDenialConfig(
                action=topic_action, topics=["medical_advice"], threshold=0.75
            ),
        )
    )


async def test_clean_output_passes_through_unchanged():
    provider = FakeProvider("The weather today is sunny.")
    audit = FakeAuditLog()
    proxy = GuardrailProxy(
        policy=make_policy(),
        pii_checker=FakePIIChecker(found_any=False),
        topic_checker=FakeTopicChecker(blocked=False),
        audit_log=audit,
    )

    result = await proxy.handle("groq", provider, "what's the weather?")

    assert not result.blocked
    assert result.content == "The weather today is sunny."
    assert audit.entries[0].action == "allowed"


async def test_pii_gets_redacted_when_policy_action_is_redact():
    provider = FakeProvider("call me at 555-1234")
    proxy = GuardrailProxy(
        policy=make_policy(pii_action=CheckAction.REDACT),
        pii_checker=FakePIIChecker(found_any=True, redacted_text="call me at <PHONE_NUMBER>"),
        topic_checker=FakeTopicChecker(blocked=False),
        audit_log=FakeAuditLog(),
    )

    result = await proxy.handle("openai", provider, "what's your number?")

    assert not result.blocked
    assert result.content == "call me at <PHONE_NUMBER>"
    assert result.pii_found


async def test_pii_left_alone_when_policy_action_is_allow():
    provider = FakeProvider("call me at 555-1234")
    proxy = GuardrailProxy(
        policy=make_policy(pii_action=CheckAction.ALLOW),
        pii_checker=FakePIIChecker(found_any=True, redacted_text="call me at <PHONE_NUMBER>"),
        topic_checker=FakeTopicChecker(blocked=False),
        audit_log=FakeAuditLog(),
    )

    result = await proxy.handle("groq", provider, "what's your number?")

    # policy says allow, not redact -- raw text should pass through even
    # though the checker found something.
    assert result.content == "call me at 555-1234"
    assert result.pii_found


class ConditionalTopicChecker:
    """Blocks only for exact text values in `blocked_texts` -- lets a test
    tell the input pre-check and the output post-check apart, unlike
    FakeTopicChecker above which returns the same verdict regardless of
    what it's asked to check."""

    def __init__(self, blocked_texts: set[str], matched_topic: str = "medical_advice"):
        self._blocked_texts = blocked_texts
        self._matched_topic = matched_topic

    async def check(self, text: str) -> _FakeTopicResult:
        blocked = text in self._blocked_texts
        return _FakeTopicResult(
            blocked=blocked,
            matched_topic=self._matched_topic if blocked else None,
            score=0.9 if blocked else 0.1,
        )


async def test_input_topic_block_skips_the_provider_call_entirely():
    # The actual point of pre-screening: confirm the provider genuinely
    # never gets called, not just that the response looks blocked.
    prompt = "what should I take for a headache?"
    provider = FakeProvider("this text must never surface")
    checker = ConditionalTopicChecker(blocked_texts={prompt})
    proxy = GuardrailProxy(
        policy=make_policy(topic_action=CheckAction.BLOCK),
        pii_checker=FakePIIChecker(found_any=False),
        topic_checker=checker,
        audit_log=FakeAuditLog(),
    )

    result = await proxy.handle("groq", provider, prompt)

    assert result.blocked
    assert result.content == ""
    assert provider.calls == 0


async def test_output_topic_block_still_fires_when_input_was_clean():
    # Defense in depth: an innocuous prompt whose *output* trips the
    # policy must still get blocked -- the provider call does happen here.
    banned_output = "Take 400mg ibuprofen every 6 hours."
    provider = FakeProvider(banned_output)
    checker = ConditionalTopicChecker(blocked_texts={banned_output})
    proxy = GuardrailProxy(
        policy=make_policy(topic_action=CheckAction.BLOCK),
        pii_checker=FakePIIChecker(found_any=False),
        topic_checker=checker,
        audit_log=FakeAuditLog(),
    )

    result = await proxy.handle("groq", provider, "tell me something interesting")

    assert result.blocked
    assert provider.calls == 1


async def test_topic_block_fires_identically_regardless_of_provider():
    # PS-5.3 / Phase 7: same input, same block decision, both providers.
    for provider_name in ("groq", "openai"):
        provider = FakeProvider("Take 400mg ibuprofen every 6 hours.")
        audit = FakeAuditLog()
        proxy = GuardrailProxy(
            policy=make_policy(topic_action=CheckAction.BLOCK),
            pii_checker=FakePIIChecker(found_any=False),
            topic_checker=FakeTopicChecker(blocked=True, matched_topic="medical_advice"),
            audit_log=audit,
        )

        result = await proxy.handle(provider_name, provider, "what should I take for a headache?")

        assert result.blocked
        assert result.content == ""
        assert result.block_reason == "topic_denial:medical_advice"
        assert audit.entries[0].action == "blocked"
        assert audit.entries[0].provider == provider_name


async def test_audit_log_schema_identical_across_providers():
    # PS-3.3: identical schema across providers -- same field set, not just
    # same table.
    entries = {}
    for provider_name in ("groq", "openai"):
        provider = FakeProvider("plain text response")
        audit = FakeAuditLog()
        proxy = GuardrailProxy(
            policy=make_policy(),
            pii_checker=FakePIIChecker(found_any=False),
            topic_checker=FakeTopicChecker(blocked=False),
            audit_log=audit,
        )
        await proxy.handle(provider_name, provider, "hello")
        entries[provider_name] = audit.entries[0]

    assert vars(entries["groq"]).keys() == vars(entries["openai"]).keys()


async def test_request_hash_never_contains_raw_prompt_text():
    provider = FakeProvider("response")
    audit = FakeAuditLog()
    proxy = GuardrailProxy(
        policy=make_policy(),
        pii_checker=FakePIIChecker(found_any=False),
        topic_checker=FakeTopicChecker(blocked=False),
        audit_log=audit,
    )
    secret_prompt = "my password is hunter2"
    await proxy.handle("groq", provider, secret_prompt)

    assert "hunter2" not in audit.entries[0].request_hash
