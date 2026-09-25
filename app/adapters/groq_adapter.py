"""
Groq adapter. Groq exposes an OpenAI-compatible /chat/completions endpoint,
so this ends up structurally near-identical to OpenAIAdapter -- base URL,
default model, and which env var holds the key are the only real
differences. That near-duplication is intentional, not an oversight: it's
what makes "add a third provider = one new file, zero changes elsewhere"
(the Phase 7 test) actually true. Collapsing these into one parametrized
class would save a few lines and cost you that property.
"""
from __future__ import annotations

import os

import httpx


class GroqAdapter:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    # llama-3.1-8b-instant was decommissioned for free/developer-tier Groq
    # accounts on 2026-08-16 -- this is what you hit earlier. Their own
    # deprecation notice points to openai/gpt-oss-20b as the replacement.
    # Despite the "openai/" prefix this is a model Groq hosts and serves
    # from api.groq.com, not a call routed to OpenAI.
    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        # Read the env var lazily (at call time via __init__), not at
        # import time -- importing this module in a test file shouldn't
        # blow up just because GROQ_API_KEY isn't set in that shell.
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model or self.DEFAULT_MODEL

    async def complete(self, prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY is not set")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        # One deliberate retry on top of the client's timeout -- not a
        # generic backoff library. This proxy sits on the hot path of every
        # chat request behind it; an unbounded retry loop here turns one
        # slow provider into a cascading latency problem for every caller.
        last_exc: Exception | None = None
        last_detail = ""
        for _attempt in range(2):
            try:
                resp = await self._client.post(
                    self.BASE_URL, json=payload, headers=headers, timeout=15.0
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as exc:
                # The specific bug this fixes: raise_for_status() gives you
                # a status code but the *body* -- which is where Groq/OpenAI
                # actually put the useful error ("model decommissioned",
                # "invalid api key", etc.) -- gets thrown away if you don't
                # capture it here. A bare "failed after 2 attempts" sends
                # whoever's debugging this on a network-connectivity hunt
                # when the real answer was sitting in the response the
                # whole time.
                last_exc = exc
                last_detail = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                last_detail = repr(exc)
        raise RuntimeError(
            f"Groq completion failed after 2 attempts -- {last_detail}"
        ) from last_exc
