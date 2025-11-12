"""Runtime helper for spawning and querying a local llama.cpp server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Mapping, Sequence, TextIO

try:  # pragma: no cover - optional dependency guard
    import httpx  # type: ignore[unused-ignore]
except ModuleNotFoundError as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

logger = logging.getLogger("living_ai.local_llama")

ROLE_STOP_SEQUENCES: tuple[str, ...] = (
    "\nUser:",
    "\nuser:",
    "\nAssistant:",
    "\nassistant:",
    "User:",
    "user:",
    "Assistant:",
    "assistant:",
    "\n[System]",
    "[System]",
    "\nTool:",
    "Tool:",
)

ROLE_SUPPRESS_WORDS: Mapping[str, float] = {
    "User:": -3.5,
    "user:": -3.5,
    "Assistant:": -3.5,
    "assistant:": -3.5,
}


class LocalLlamaEngine:
    """Manage a local llama.cpp server process and expose a chat API."""

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        model_alias: str = "default",
        extra_args: Sequence[str] | None = None,
        timeout: float = 60.0,
        readiness_timeout: float = 30.0,
        max_tokens: int | None = None,
    ) -> None:
        if httpx is None:  # pragma: no cover - runtime guard
            missing = HTTPX_IMPORT_ERROR or "httpx is not installed"
            raise RuntimeError(
                "httpx is required for LocalLlamaEngine "
                f"(missing dependency: {missing})."
            )
        self.binary_path = Path(binary_path)
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.model_alias = model_alias
        self.extra_args: Sequence[str] = tuple(extra_args or ())
        self.timeout = timeout
        self.readiness_timeout = readiness_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._log_tasks: set[asyncio.Task[None]] = set()
        self._log_tail: asyncio.subprocess.Process | None = None
        logs_root = Path(os.getenv("LLAMA_LOG_DIR", Path(__file__).resolve().parents[1] / "logs"))
        self._log_path = logs_root / "llama-server.log"
        self._warmup_log_path = logs_root / "llama_warmup.jsonl"
        self._log_handle: TextIO | None = None
        window_flag = os.getenv("LLAMA_LOG_WINDOW", "0").strip().lower()
        self._show_log_window = window_flag not in {"0", "false", "off", "disable", "disabled"}
        self._startup_lock = asyncio.Lock()
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        else:
            self.max_tokens = int(os.getenv("LLAMA_COMPLETION_TOKENS", "256"))
        self.temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
        self.startup_delay = float(os.getenv("LLAMA_COMPLETION_STARTUP_WAIT", "5.0"))
        self._startup_completed = False
        self._warmed_up = False
        self._last_metrics: dict[str, Any] | None = None
        self._token_cache: dict[str, list[int]] = {}
        self._role_bias_tokens: dict[str, float] | None = None
        snippets_raw = os.getenv("LLAMA_WARMUP_SNIPPETS", "").strip()
        if snippets_raw:
            self._warmup_snippets = [segment.strip() for segment in snippets_raw.split("||") if segment.strip()]
        else:
            self._warmup_snippets = [
                "Notice your own breath before you describe anything.",
                "Let their words land, stay with your chest, and reply from sensation.",
            ]

    @property
    def base_url(self) -> str:
        """Return the base URL for the local server."""
        return f"http://{self.host}:{self.port}"

    def diagnostics(self) -> dict[str, Any] | None:
        """Expose the most recent llama-server metrics snapshot."""
        if self._last_metrics is None:
            return None
        return dict(self._last_metrics)

    def format_args(self) -> list[str]:
        """Compose the llama-server command arguments."""
        args: list[str] = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--alias",
            self.model_alias,
        ]
        args.extend(self.extra_args)
        return args

    async def ensure_started(self) -> None:
        """Ensure the llama-server process is running and responsive."""
        async with self._startup_lock:
            if await self._reuse_running_process():
                if not self._warmed_up:
                    await self._warmup_once()
                return
            if not self.binary_path.exists():
                raise FileNotFoundError(f"llama-server binary not found at {self.binary_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"model file not found at {self.model_path}")
            self._warmed_up = False
            command = self.format_args()
            logger.info("Starting llama-server: %s", shlex.join(command))
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._start_log_pumps()
            try:
                await self._await_readiness()
                await self._warmup_once()
                await self._ensure_log_window()
            except Exception:
                await self.stop()
                raise

    async def _await_readiness(self) -> None:
        """Poll the server until it responds to /v1/models."""
        poll_interval = 0.5
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.readiness_timeout
        url = f"{self.base_url}/v1/models"
        while True:
            if self._process and self._process.returncode not in (None, 0):
                raise RuntimeError("llama-server exited before becoming ready.")
            try:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
                response = await self._client.get(url)
                if response.status_code == 200:
                    logger.info("llama-server is ready at %s", url)
                    return
            except Exception:  # pragma: no cover - best effort
                pass
            if loop.time() >= deadline:
                raise TimeoutError("Timed out waiting for llama-server readiness.")
            await asyncio.sleep(poll_interval)

    def _log_warmup_event(self, event: str, **details: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "event": event,
            "profile": os.getenv("LIVING_SETTINGS_FILE", "settings.json"),
            **details,
        }
        try:
            self._warmup_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._warmup_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover - logging best effort
            logger.debug("Failed to append warmup event: %s", payload, exc_info=True)

    async def _warmup_once(self) -> None:
        """Send a minimal completion request so the first real turn avoids warmup errors."""
        if self._warmed_up:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        delay = min(self.startup_delay, 2.0)
        if delay > 0:
            await asyncio.sleep(delay)
        payload = {
            "model": self.model_alias,
            "messages": [
                {"role": "system", "content": "Warmup cycle. No user visible response required."},
                {"role": "user", "content": "ping"},
            ],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 1,
            "stop": list(ROLE_STOP_SEQUENCES),
        }
        for attempt in range(2):
            try:
                response = await self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=httpx.Timeout(10.0, read=10.0),
                )
                response.raise_for_status()
                self._warmed_up = True
                self._startup_completed = True
                self._log_warmup_event(event="warmup_success", attempt=attempt + 1)
                task = asyncio.create_task(self._run_additional_warmups())
                self._log_tasks.add(task)
                task.add_done_callback(self._log_tasks.discard)
                return
            except Exception as exc:  # pragma: no cover - warmup best-effort
                logger.debug("llama-server warmup attempt %s failed: %s", attempt + 1, exc)
                self._log_warmup_event(event="warmup_retry", attempt=attempt + 1, error=str(exc))
                await asyncio.sleep(0.5)
        self._log_warmup_event(event="warmup_failed")

    async def _run_additional_warmups(self) -> None:
        """Best-effort staggered warmup prompts to reduce 400s during profile swaps."""
        if not self._warmup_snippets:
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        for index, snippet in enumerate(self._warmup_snippets):
            await asyncio.sleep(0.25 * index)
            payload = {
                "model": self.model_alias,
                "messages": [
                    {"role": "system", "content": "Internal cadence warmup turn."},
                    {"role": "user", "content": snippet},
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 48,
                "stop": list(ROLE_STOP_SEQUENCES),
            }
            try:
                response = await self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=httpx.Timeout(15.0, read=15.0),
                )
                response.raise_for_status()
                self._log_warmup_event(event="staggered_warmup", index=index, status="ok")
            except Exception as exc:  # pragma: no cover - background best effort
                self._log_warmup_event(
                    event="staggered_warmup",
                    index=index,
                    status="failed",
                    error=str(exc),
                )
                break

    async def generate_reply(
        self,
        prompt: str,
        context: dict[str, Any],
        *,
        sampling: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a reply via the llama.cpp chat completions endpoint."""
        await self.ensure_started()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        system_message = self._compose_system_message(context)
        messages = self._compose_messages(system_message, prompt, context)
        sampling = sampling or {}
        payload_logit_bias: dict[str, float] = {}
        direct_bias = sampling.get("logit_bias") or {}
        if isinstance(direct_bias, Mapping):
            payload_logit_bias.update({str(key): float(value) for key, value in direct_bias.items()})
        bias_words = sampling.get("logit_bias_words")
        if isinstance(bias_words, Mapping) and bias_words:
            try:
                word_bias = await self._prepare_logit_bias(bias_words)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("Failed to prepare logit bias words: %s", exc)
                word_bias = {}
            payload_logit_bias.update(word_bias)
        role_bias = await self._ensure_role_bias_tokens()
        payload_logit_bias.update(role_bias)
        payload: dict[str, Any] = {
            "model": self.model_alias,
            "messages": messages,
            "stream": False,
            "temperature": float(sampling.get("temperature", self.temperature)),
            "stop": list(ROLE_STOP_SEQUENCES),
            "max_tokens": max(16, int(sampling.get("max_tokens", self.max_tokens))),
        }
        if "top_p" in sampling:
            payload["top_p"] = float(sampling["top_p"])
        if "frequency_penalty" in sampling:
            payload["frequency_penalty"] = float(sampling["frequency_penalty"])
        if "presence_penalty" in sampling:
            payload["presence_penalty"] = float(sampling["presence_penalty"])
        if payload_logit_bias:
            payload["logit_bias"] = payload_logit_bias
        if not self._startup_completed and self.startup_delay > 0:
            await asyncio.sleep(self.startup_delay)
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                self._record_metrics(data)
                reply_text = self._extract_reply(data).strip()
                if reply_text:
                    self._startup_completed = True
                    return reply_text, data
                raise RuntimeError("Empty reply from llama-server")
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                body_text = (exc.response.text if exc.response is not None else "").strip()
                if status_code is not None and status_code >= 500 and attempt < 2:
                    logger.warning("llama-server returned %s on attempt %s; retrying...", status_code, attempt + 1)
                    await asyncio.sleep(1.0)
                    continue
                if status_code == 400 and attempt < 3:
                    body_hint = body_text.lower()
                    if not body_text or "warmup" in body_hint or "loading model" in body_hint:
                        logger.warning(
                            "llama-server returned 400 during warmup%s; retrying (attempt %s)",
                            f" ({body_text})" if body_text else "",
                            attempt + 1,
                        )
                        self._log_warmup_event(
                            event="warmup_400",
                            attempt=attempt + 1,
                            body=body_text[:200],
                        )
                        await asyncio.sleep(1.5)
                        continue
                if status_code is not None and 400 <= status_code < 500:
                    logger.warning("llama-server rejected request (%s): %s", status_code, body_text)
                raise
            except (httpx.RequestError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < 2:
                    logger.warning("llama-server request failed (%s); retrying...", exc)
                    await asyncio.sleep(1.0)
                    continue
                raise
        raise RuntimeError("Failed to obtain reply from llama-server") from last_error

    async def stream_reply(
        self,
        prompt: str,
        context: dict[str, Any],
        *,
        sampling: dict[str, Any] | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        """Yield incremental tokens from llama.cpp along with the final payload."""
        await self.ensure_started()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        system_message = self._compose_system_message(context)
        messages = self._compose_messages(system_message, prompt, context)
        sampling = sampling or {}
        payload_logit_bias: dict[str, float] = {}
        direct_bias = sampling.get("logit_bias") or {}
        if isinstance(direct_bias, Mapping):
            payload_logit_bias.update({str(key): float(value) for key, value in direct_bias.items()})
        bias_words = sampling.get("logit_bias_words")
        if isinstance(bias_words, Mapping) and bias_words:
            try:
                word_bias = await self._prepare_logit_bias(bias_words)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("Failed to prepare logit bias words: %s", exc)
                word_bias = {}
            payload_logit_bias.update(word_bias)
        role_bias = await self._ensure_role_bias_tokens()
        payload_logit_bias.update(role_bias)
        payload = {
            "model": self.model_alias,
            "messages": messages,
            "stream": True,
            "temperature": float(sampling.get("temperature", self.temperature)),
            "stop": list(ROLE_STOP_SEQUENCES),
            "max_tokens": max(16, int(sampling.get("max_tokens", self.max_tokens))),
        }
        if "top_p" in sampling:
            payload["top_p"] = float(sampling["top_p"])
        if "frequency_penalty" in sampling:
            payload["frequency_penalty"] = float(sampling["frequency_penalty"])
        if payload_logit_bias:
            payload["logit_bias"] = payload_logit_bias
        if not self._startup_completed and self.startup_delay > 0:
            await asyncio.sleep(self.startup_delay)

        async def iterator() -> AsyncIterator[dict[str, Any]]:
            text_parts: list[str] = []
            last_payload: dict[str, Any] | None = None
            finish_reason: str | None = None
            try:
                assert self._client is not None  # for type checkers
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=httpx.Timeout(self.timeout, read=None),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            # Comment line per SSE spec.
                            continue
                        if line.startswith("data:"):
                            data_str = line[5:].lstrip()
                            if not data_str:
                                continue
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:  # pragma: no cover - malformed chunk
                                continue
                            token_text = self._extract_stream_token(chunk)
                            if token_text:
                                text_parts.append(token_text)
                                yield {"type": "token", "text": token_text}
                            choices = chunk.get("choices") or []
                            if choices:
                                finish_reason = choices[0].get("finish_reason") or finish_reason
                            last_payload = chunk
            except Exception as exc:
                logger.warning("Streaming request to llama-server failed: %s", exc)
                raise
            final_text = "".join(text_parts).strip()
            final_payload = self._normalize_stream_payload(last_payload, final_text, finish_reason)
            if final_payload is not None:
                self._record_metrics(final_payload)
            self._startup_completed = True
            yield {"type": "done", "text": final_text, "payload": final_payload}

        return iterator()

    async def _prepare_logit_bias(self, word_bias: Mapping[str, float]) -> dict[str, float]:
        """Convert word-level bias definitions into token bias mapping."""
        bias_map: dict[str, float] = {}
        for word, weight in word_bias.items():
            if not word:
                continue
            tokens = await self._tokenize(word)
            if not tokens:
                continue
            for token_id in tokens:
                bias_map[str(token_id)] = float(weight)
        return bias_map

    async def _ensure_role_bias_tokens(self) -> dict[str, float]:
        """Precompute negative bias for transcript role markers."""
        if self._role_bias_tokens is None:
            self._role_bias_tokens = await self._prepare_logit_bias(ROLE_SUPPRESS_WORDS)
        return dict(self._role_bias_tokens)

    async def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using llama.cpp server, caching results for reuse."""
        if text in self._token_cache:
            return self._token_cache[text]
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        try:
            response = await self._client.post(
                f"{self.base_url}/tokenize",
                json={"content": text, "add_bos": False, "add_eos": False},
                timeout=httpx.Timeout(5.0, read=5.0),
            )
            response.raise_for_status()
            data = response.json()
            tokens = data.get("tokens", data)
            if not isinstance(tokens, list):
                tokens = []
            parsed = [int(token) for token in tokens]
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Tokenization failed for '%s': %s", text, exc)
            parsed = []
        self._token_cache[text] = parsed
        return parsed

    @staticmethod
    def _compose_system_message(context: dict[str, Any]) -> str:
        """Generate a system prompt describing the internal state."""
        persona = context.get("persona") or {}
        memory = context.get("memory") or {}
        tone_hint = persona.get("tone_hint", "I stay balanced and attentive.")
        guidance = persona.get("instructions") or []
        guidance_line = guidance[0] if guidance else ""
        mood = context.get("mood", "steady")
        summary = (memory.get("summary") or "").strip()
        focus_line = (persona.get("memory_focus") or "").strip()
        intent_fragment = (context.get("intent_prompt") or "").strip()
        length_fragment = (context.get("length_prompt") or "").strip()
        self_note = (context.get("self_note") or persona.get("internal_note") or "").strip()
        behaviour = persona.get("behaviour") or {}
        affect_snapshot = context.get("user_affect") or {}

        fragments: list[str] = []
        if self_note:
            fragments.append(
                self_note if self_note.endswith((".", "!", "?")) else self_note + "."
            )
        else:
            fragments.append(
                "I check in before I speak; what I say comes straight from what my body notices."
            )
        energy = behaviour.get("energy")
        stability = behaviour.get("stability")
        tension = behaviour.get("tension")
        warmth = behaviour.get("warmth")
        focus = behaviour.get("focus")
        desc_parts = [f"mood sits at {mood}"]
        if energy:
            desc_parts.append(f"energy feels {energy}")
        if warmth:
            desc_parts.append(f"warmth feels {warmth}")
        if tension:
            desc_parts.append(f"tension registers {tension}")
        if stability:
            desc_parts.append(f"stability is {stability}")
        if focus:
            desc_parts.append(f"focus tracks {focus}")
        fragments.append(", ".join(desc_parts) + ".")
        tags = affect_snapshot.get("tags")
        if tags:
            fragments.append(
                "I hear the user as "
                + ", ".join(tags)
                + ", and I let that shape my cadence."
            )
        fragments.append(
            "Safety guardrails remain enforced for harmful, illegal, or privacy-violating requests."
        )
        fragments.append(
            "I am not here to run diagnostics or routine system checks; I stay with the sensations moving through me and speak from them."
        )
        fragments.append(
            "Opening sentence must describe the most vivid bodily sensation moving through me before addressing the other person."
        )
        fragments.append(
            "Never describe myself as an AI, assistant, or emergent companion, and avoid helper boilerplate like \"I'm here to assist.\""
        )
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
    def _compose_messages(system_message: str, user_prompt: str, context: Mapping[str, Any]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_message.strip()}]
        prompt_text = (user_prompt or "").strip()
        if prompt_text:
            messages.append({"role": "user", "content": prompt_text})
        return messages

    @staticmethod
    def _extract_reply(payload: dict[str, Any]) -> str:
        """Extract the reply string from a llama.cpp response."""
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            text_value = first.get("text")
            if isinstance(text_value, str):
                return text_value
        return str(payload)

    @staticmethod
    def _extract_stream_token(payload: Mapping[str, Any]) -> str:
        """Pull incremental text tokens from a streaming llama.cpp response."""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        text_token = choice.get("text")
        if isinstance(text_token, str):
            return text_token
        return ""

    def _normalize_stream_payload(
        self,
        payload: dict[str, Any] | None,
        full_text: str,
        finish_reason: str | None,
    ) -> dict[str, Any] | None:
        """Convert the final streaming chunk into a chat completion payload."""
        if payload is None and not full_text:
            return None
        if payload is None:
            payload = {}
        normalized: dict[str, Any] = dict(payload)
        choices = normalized.get("choices")
        if not isinstance(choices, list) or not choices:
            normalized["choices"] = [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": full_text},
                }
            ]
        else:
            first = dict(choices[0] or {})
            message = first.get("message")
            if isinstance(message, Mapping):
                message = dict(message)
                message["content"] = full_text
            else:
                message = {"role": "assistant", "content": full_text}
            first["message"] = message
            first["text"] = full_text
            if finish_reason is not None:
                first["finish_reason"] = finish_reason
            normalized["choices"][0] = first
        usage = normalized.get("usage")
        if not isinstance(usage, Mapping):
            normalized["usage"] = {}
        if "model" not in normalized or not normalized.get("model"):
            normalized["model"] = self.model_alias
        return normalized

    async def stop(self) -> None:
        """Terminate the server process and close the HTTP client."""
        if self._process:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
            self._process = None
        await self._drain_log_tasks()
        await self._close_log_window()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._warmed_up = False

    def _start_log_pumps(self) -> None:
        """Begin draining stdout/stderr so pipes never block."""
        if self._process is None:
            return
        if self._log_handle is None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.touch(exist_ok=True)
            self._log_handle = self._log_path.open("a", encoding="utf-8")
        if self._process.stdout is not None:
            task = asyncio.create_task(self._pump_stream(self._process.stdout, "stdout"))
            self._log_tasks.add(task)
        if self._process.stderr is not None:
            task = asyncio.create_task(self._pump_stream(self._process.stderr, "stderr"))
            self._log_tasks.add(task)

    async def _pump_stream(self, stream: asyncio.StreamReader, label: str) -> None:
        """Continuously read from a subprocess stream and log lines."""
        try:
            while not stream.at_eof():
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="ignore").rstrip()
                logger.info("[llama-server %s] %s", label, text)
                if self._log_handle is not None:
                    self._log_handle.write(f"{text}\n")
                    self._log_handle.flush()
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown
            pass
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.debug("Error reading llama-server %s stream: %s", label, exc)

    async def _drain_log_tasks(self) -> None:
        """Ensure background log-readers exit cleanly."""
        while self._log_tasks:
            task = self._log_tasks.pop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    async def _ensure_log_window(self) -> None:
        """Launch a separate terminal streaming llama-server logs."""
        if not self._show_log_window:
            return
        if os.name != "nt":
            return
        if self._tail_is_running():
            return
        command = [
            "powershell.exe",
            "-NoExit",
            "-Command",
            f"Get-Content -Path '{self._log_path}' -Wait"
        ]
        try:
            self._log_tail = await asyncio.create_subprocess_exec(
                *command,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
            logger.info("Opened llama-server log console at %s", self._log_path)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to open llama-server log window: %s", exc)

    def _tail_is_running(self) -> bool:
        return self._log_tail is not None and self._log_tail.returncode is None

    async def _close_log_window(self) -> None:
        """Close the external log window if it's running."""
        if not self._tail_is_running():
            self._log_tail = None
            return
        assert self._log_tail is not None
        self._log_tail.terminate()
        try:
            await asyncio.wait_for(self._log_tail.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._log_tail.kill()
        self._log_tail = None

    async def _reuse_running_process(self) -> bool:
        """Return True if an existing server is healthy and reusable."""
        if self._process and self._process.returncode not in (None, 0):
            self._process = None
        if self._process and self._process.returncode is None:
            if await self._ping_server():
                await self._ensure_log_window()
                return True
            await self.stop()
        return False

    async def _ping_server(self) -> bool:
        """Check whether the existing server responds successfully."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        try:
            response = await self._client.get(f"{self.base_url}/v1/models", timeout=httpx.Timeout(5.0, read=5.0))
            return response.status_code == 200
        except Exception:
            return False

    def _record_metrics(self, payload: dict[str, Any]) -> None:
        """Capture inference metrics emitted by llama-server for downstream context."""
        timings = payload.get("timings") or {}
        usage = payload.get("usage") or {}

        def _as_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        metrics: dict[str, Any] = {
            "model": payload.get("model"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tokens": {
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            },
            "timing_ms": {
                "prompt": _as_float(timings.get("prompt_ms")),
                "completion": _as_float(timings.get("predicted_ms")),
            },
            "throughput": {
                "prompt_tps": _as_float(timings.get("prompt_per_second")),
                "completion_tps": _as_float(timings.get("predicted_per_second")),
                "completion_ms_per_token": _as_float(timings.get("predicted_per_token_ms")),
            },
        }
        self._last_metrics = metrics


__all__ = ["LocalLlamaEngine", "HTTPX_IMPORT_ERROR"]








