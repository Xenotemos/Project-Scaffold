"""Async client for interacting with the living_llm bridge service."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

try:  # pragma: no cover - optional dependency guard
    import httpx  # type: ignore[unused-ignore]
except ModuleNotFoundError as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover
    import httpx

logger = logging.getLogger("living_ai.llm_client")


class LivingLLMClient:
    """Lightweight wrapper around the living_llm `/chat` endpoint."""

    RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, endpoint: str, *, timeout: float = 15.0, max_retries: int = 2) -> None:
        if httpx is None:  # pragma: no cover - runtime guard
            missing = HTTPX_IMPORT_ERROR or "httpx is not installed"
            raise RuntimeError(
                "httpx is required to use LivingLLMClient. "
                f"Install httpx to enable the living_llm bridge (missing import: {missing})."
            )
        self._endpoint = endpoint.rstrip("/") or "http://localhost:8001/chat"
        self._client = httpx.AsyncClient(timeout=timeout)
        self._max_retries = max(0, int(max_retries))
        self._base_backoff = 0.4

    async def generate_reply(self, prompt: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Send a prompt with context and return the extracted reply plus raw payload."""
        payload = {
            "prompt": prompt,
            "context_override": context,
            "stream": False,
        }
        logger.debug("Dispatching prompt to living_llm endpoint '%s'", self._endpoint)
        attempt = 0
        last_exc: Exception | None = None
        backoff = self._base_backoff
        while True:
            try:
                response = await self._client.post(self._endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                reply_text = self._extract_reply(data.get("result"))
                return reply_text, data
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code in self.RETRYABLE_STATUS
                logger.warning(
                    "LLM bridge returned %s (retryable=%s attempt=%s/%s)",
                    status_code,
                    retryable,
                    attempt + 1,
                    self._max_retries + 1,
                )
                if retryable and attempt < self._max_retries:
                    last_exc = exc
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 4.0)
                    attempt += 1
                    continue
                raise
            except httpx.RequestError as exc:
                logger.warning(
                    "LLM bridge request error (attempt %s/%s): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 4.0)
                    attempt += 1
                    continue
                raise
        # Should never reach here
        raise RuntimeError("LLM bridge failed") from last_exc

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    @staticmethod
    def _extract_reply(result: Any) -> str:
        """Normalise the result payload into a reply string."""
        if isinstance(result, dict):
            for key in ("reply", "text", "message", "content", "output"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return str(result)
        if result is None:
            return ""
        return str(result)


__all__ = ["LivingLLMClient", "HTTPX_IMPORT_ERROR"]
