"""OpenAI adapter -- same shape as GroqAdapter, see that file's docstring."""
from __future__ import annotations

import os

import httpx


class OpenAIAdapter:
    BASE_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or self.DEFAULT_MODEL

    async def complete(self, prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

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
                last_exc = exc
                last_detail = f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                last_exc = exc
                last_detail = repr(exc)
        raise RuntimeError(
            f"OpenAI completion failed after 2 attempts -- {last_detail}"
        ) from last_exc
