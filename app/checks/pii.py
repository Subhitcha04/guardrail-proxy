"""
PII detection + redaction via Presidio.

Presidio's analyzer/anonymizer are synchronous and CPU-bound under the hood
(they run a spaCy NER pass). Calling them directly inside an `async def`
route blocks the event loop for every other in-flight request -- pitfall #2
in the README. `check()` wraps the synchronous work in `asyncio.to_thread`
so the event loop stays free while the NER pass runs.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Presidio's default config loads ~15-20 recognizers simultaneously,
# several of them locale-specific ID patterns (IN_PAN, IN_AADHAAR,
# US_ITIN, UK_NHS, SG_NRIC_FIN, ...) that stay active regardless of the
# text's actual locale. Started as a denylist (LOCATION, DATE_TIME) after
# "France"/"Paris" and "quarterly" got redacted out of ordinary sentences.
# Kept finding more: IN_PAN then redacted the plain English words
# "contacting" and "additional" out of a support-ticket template -- three
# separate false-positive categories inside a handful of test completions
# makes denylisting a losing game against recognizers we haven't even
# inspected yet. Switched to an allowlist instead: only entity types
# verified correct against real model output, across multiple runs, stay
# enabled. Extend deliberately -- CREDIT_CARD, US_SSN, IBAN_CODE are
# reasonable additions if the spec calls for them, but they're untested
# here, so they're not in by default.
ALLOWED_ENTITY_TYPES = frozenset({"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"})


@dataclass
class PIIFinding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class PIICheckResult:
    redacted_text: str
    findings: list[PIIFinding] = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return len(self.findings) > 0


class PIIChecker:
    """
    Owns one AnalyzerEngine + AnonymizerEngine instance. Both load spaCy's
    `en_core_web_lg` model into memory (~600MB) on first use -- construct
    this once at startup (see main.py's lifespan), never per-request. This
    is the same "load once, not per-request" rule as the embedding model in
    topic_denial.py, just easier to miss here because Presidio hides the
    model loading inside its own constructor.
    """

    def __init__(
        self,
        analyzer: AnalyzerEngine | None = None,
        anonymizer: AnonymizerEngine | None = None,
    ) -> None:
        self._analyzer = analyzer or AnalyzerEngine()
        self._anonymizer = anonymizer or AnonymizerEngine()

    def _check_sync(self, text: str, language: str = "en") -> PIICheckResult:
        # entities= passed directly to analyze() (not post-filtered): with
        # an allowlist this also means the excluded recognizers -- LOCATION,
        # DATE_TIME, IN_PAN, and everything else outside the allowlist --
        # never run at all, not just that their output gets discarded.
        results = self._analyzer.analyze(
            text=text, language=language, entities=list(ALLOWED_ENTITY_TYPES)
        )
        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        findings = [
            PIIFinding(entity_type=r.entity_type, start=r.start, end=r.end, score=r.score)
            for r in results
        ]
        return PIICheckResult(redacted_text=anonymized.text, findings=findings)

    async def check(self, text: str, language: str = "en") -> PIICheckResult:
        return await asyncio.to_thread(self._check_sync, text, language)
