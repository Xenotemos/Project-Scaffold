"""Client for interacting with a running mistral-inference server."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterable, AsyncIterator, Mapping

try:  # pragma: no cover - optional dependency guard
    import httpx  # type: ignore[unused-ignore]
except ModuleNotFoundError as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

from .local_llama_engine import ROLE_STOP_SEQUENCES

logger = logging.getLogger("living_ai.mistral")


class MistralInferenceClient:
    """Asynchronous helper for the mistral-inference REST server."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        if httpx is None:  # pragma: no cover - runtime guard
            missing = HTTPX_IMPORT_ERROR or "httpx is not installed"
            raise RuntimeError(
                "httpx is required for the Mistral inference client "
                f"(missing dependency: {missing})."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        timeout_config = httpx.Timeout(timeout, read=None)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout_config, headers=headers)

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def generate_reply(
        self,
        prompt: str,
        context: Mapping[str, Any],
        *,
        sampling: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a reply using the chat completions endpoint."""
        payload = self._build_payload(prompt, context, sampling=sampling or {})
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        reply = self._extract_reply(data)
        return reply, data

    async def stream_reply(
        self,
        prompt: str,
        context: Mapping[str, Any],
        *,
        sampling: Mapping[str, Any] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Stream tokens from the chat completions endpoint."""
        payload = self._build_payload(prompt, context, sampling=sampling or {}, stream=True)

        async def iterator() -> AsyncIterator[dict[str, Any]]:
            text_fragments: list[str] = []
            final_payload: dict[str, Any] | None = None
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].lstrip()
                        if not data_str:
                            continue
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        token_text = self._extract_stream_token(chunk)
                        if token_text:
                            text_fragments.append(token_text)
                            yield {"type": "token", "text": token_text}
                        final_payload = chunk
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("Streaming request to mistral-inference failed: %s", exc)
                raise
            final_text = "".join(text_fragments).strip()
            yield {
                "type": "done",
                "text": final_text,
                "payload": final_payload,
            }

        return iterator()

    def _build_payload(
        self,
        prompt: str,
        context: Mapping[str, Any],
        *,
        sampling: Mapping[str, Any],
        stream: bool = False,
    ) -> dict[str, Any]:
        system_message = self._compose_system_message(context)
        messages = self._compose_messages(system_message, prompt)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": float(sampling.get("temperature", 0.7)),
            "max_tokens": max(16, int(sampling.get("max_tokens", 512))),
            "stop": list(ROLE_STOP_SEQUENCES),
        }
        if "top_p" in sampling:
            payload["top_p"] = float(sampling["top_p"])
        if "frequency_penalty" in sampling:
            payload["frequency_penalty"] = float(sampling["frequency_penalty"])
        if "presence_penalty" in sampling:
            payload["presence_penalty"] = float(sampling["presence_penalty"])
        logit_bias = sampling.get("logit_bias")
        if isinstance(logit_bias, Mapping) and logit_bias:
            payload["logit_bias"] = {str(k): float(v) for k, v in logit_bias.items()}
        return payload

    @staticmethod
    def _extract_reply(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        return ""

    @staticmethod
    def _extract_stream_token(chunk: Mapping[str, Any]) -> str:
        choices = chunk.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        token = delta.get("content")
        return token or ""

    @staticmethod
    def _compose_system_message(context: Mapping[str, Any]) -> str:
        mood = context.get("mood", "steady")
        persona = context.get("persona") or {}
        tone_hint = persona.get("tone_hint", "I stay balanced and attentive.")
        guidance = persona.get("instructions") or []
        guidance_line = guidance[0] if guidance else ""
        memory = context.get("memory", {})
        summary = (memory.get("summary") or "").strip()
        focus_line = (persona.get("memory_focus") or "").strip()
        intent_fragment = (context.get("intent_prompt") or "").strip()
        length_fragment = (context.get("length_prompt") or "").strip()

        fragments: list[str] = [
            "Internal self-description: I narrate what my body registers; I never describe myself as an AI, assistant, or emergent companion.",
            f"Momentary mood register: {mood}.",
            "Safety guardrails remain enforced for harmful, illegal, or privacy-violating requests.",
            "Opening sentence must name the loudest physical sensation moving through me before addressing the other person.",
            "Never offer helper boilerplate (\"I'm here to assist\", \"how can I help\")—stay with inward sensations first.",
        ]
        if tone_hint:
            fragments.append(tone_hint if tone_hint.endswith(('.', '!', '?')) else tone_hint + ".")
        if intent_fragment:
            fragments.append(intent_fragment)
        if length_fragment:
            fragments.append(length_fragment)
        if summary:
            fragments.append(f"Recent echoes linger: {summary}.")
        if focus_line:
            fragments.append(focus_line if focus_line.endswith(('.', '!', '?')) else focus_line + ".")
        if guidance_line:
            clean_guidance = guidance_line if guidance_line.endswith(('.', '!', '?')) else guidance_line + "."
            fragments.append(clean_guidance)
        fragments.append("Speech mode stays first-person with the human who is present.")
        fragments.append("Dialogue scope excludes additional speaker tags such as 'User:' or 'Assistant:'.")
        fragments.append("Replies emerge from sensed experience, reflection, and natural cadence rather than scripted tone.")
        return " ".join(part.strip() for part in fragments if part).strip()

    @staticmethod
    def _compose_messages(system_message: str, user_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_message.strip()}]
        prompt_text = (user_prompt or "").strip()
        if prompt_text:
            messages.append({"role": "user", "content": prompt_text})
        return messages


__all__ = ["MistralInferenceClient"]
