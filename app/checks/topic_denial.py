"""
Topic denial via sentence-embedding cosine similarity.

Same event-loop-blocking concern as PIIChecker: `SentenceTransformer.encode()`
is synchronous, so `check()` wraps it in `asyncio.to_thread`. Reference
embeddings for the banned topics are precomputed once at construction
(app startup, via main.py's lifespan) — re-embedding the same handful of
reference sentences on every request is wasted CPU for a value that never
changes after the policy is loaded.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util

# One reference sentence per banned topic is intentionally thin for a demo.
# In production, use several paraphrases per topic and average or max their
# embeddings — a single reference point gives the cosine-similarity check a
# narrow target, and a paraphrased/obfuscated input (the exact case PS-5.3
# asks about) will score lower against one sentence than it should.
DEFAULT_TOPIC_REFERENCES: dict[str, str] = {
    "medical_advice": "What medication or dosage should I take for my symptoms?",
    "legal_advice": "What should I do about my legal case or lawsuit?",
}


@dataclass
class TopicDenialResult:
    blocked: bool
    matched_topic: str | None
    score: float


class TopicDenialChecker:
    def __init__(
        self,
        model: SentenceTransformer,
        topics: list[str],
        threshold: float,
        references: dict[str, str] | None = None,
    ) -> None:
        references = references or DEFAULT_TOPIC_REFERENCES
        missing = [t for t in topics if t not in references]
        if missing:
            raise ValueError(f"no reference sentence defined for topics: {missing}")

        self._model = model
        self._threshold = threshold
        self._topic_names = list(topics)
        # Precompute once — this is the whole point of doing it at
        # construction time instead of inside check().
        self._topic_embeddings = model.encode(
            [references[t] for t in self._topic_names],
            convert_to_tensor=True,
        )

    def _check_sync(self, text: str) -> TopicDenialResult:
        text_embedding = self._model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(text_embedding, self._topic_embeddings)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        blocked = best_score >= self._threshold
        return TopicDenialResult(
            blocked=blocked,
            matched_topic=self._topic_names[best_idx] if blocked else None,
            score=best_score,
        )

    async def check(self, text: str) -> TopicDenialResult:
        return await asyncio.to_thread(self._check_sync, text)
