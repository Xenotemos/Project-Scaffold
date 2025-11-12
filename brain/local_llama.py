"""Runtime helper for spawning and querying a local llama.cpp server."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

try:  # pragma: no cover - optional dependency guard
    import httpx  # type: ignore[unused-ignore]
except ModuleNotFoundError as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

logger = logging.getLogger("living_ai.local_llama")


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
        timeout: float = 30.0,
        readiness_timeout: float = 30.0,
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
        self._log_handle: TextIO | None = None
        self._show_log_window = os.getenv("LLAMA_LOG_WINDOW", "1") != "0"
        self._startup_lock = asyncio.Lock()
        self._last_metrics: dict[str, Any] | None = None

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
                return
            if not self.binary_path.exists():
                raise FileNotFoundError(f"llama-server binary not found at {self.binary_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"model file not found at {self.model_path}")
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

    async def generate_reply(self, prompt: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Send a chat completion request and return the reply text."""
        await self.ensure_started()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, read=None))
        system_message = self._compose_system_message(context)
        payload = {
            "model": self.model_alias,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        self._record_metrics(data)
        reply_text = self._extract_reply(data)
        return reply_text, data

    @staticmethod
    def _compose_system_message(context: dict[str, Any]) -> str:
        """Generate a system prompt describing the internal state."""
        persona = context.get("persona") or {}
        mood = context.get("mood", "neutral")
        memory = context.get("memory", {})
        summary = persona.get("memory_summary") or memory.get("summary", "quiet observations.")
        working_items: Iterable[str] = memory.get("working") or ()
        focus_line = persona.get("memory_focus")
        if not focus_line:
            focus_line = (
                "Current focus: "
                + ("; ".join(str(item) for item in working_items) or "no single anchor.")
            )
        long_term = memory.get("long_term") or []
        metrics = context.get("llama_metrics") or {}
        tone = persona.get("tone", mood)
        signals = persona.get("signals") or []
        guidance = persona.get("guidance") or []
        inner_voice = persona.get("inner_voice", "")
        signal_sentence = " ".join(signals) if signals else "Energy feels steady and receptive."
        guidance_sentence = (
            " ".join(guidance) if guidance else "Lean into a collaborative, empathetic tone."
        )

        metrics_chunks: list[str] = []
        if isinstance(metrics, dict):
            timestamp = metrics.get("timestamp")
            timing_ms = metrics.get("timing_ms") or {}
            throughput = metrics.get("throughput") or {}
            completion_tps = throughput.get("completion_tps")
            completion_latency = timing_ms.get("completion")
            completion_ms_tok = throughput.get("completion_ms_per_token")
            if completion_tps:
                metrics_chunks.append(f"completion throughput {completion_tps:.1f} tok/s")
            if completion_latency:
                metrics_chunks.append(f"completion latency {completion_latency:.0f} ms")
            if completion_ms_tok:
                metrics_chunks.append(f"{completion_ms_tok:.1f} ms per token")
            if timestamp:
                metrics_chunks.append(f"captured at {timestamp}")
        if not metrics_chunks:
            metrics_chunks.append("metrics unavailable")
        metrics_summary = "; ".join(metrics_chunks)

        long_term_items = "; ".join(
            entry.get("content", str(entry)) if isinstance(entry, dict) else str(entry)
            for entry in long_term
        )
        long_term_phrase = f"Long-term echoes: {long_term_items}." if long_term_items else ""

        internal_cues = ", ".join(f"{key}={value}" for key, value in behaviour.items()) or "energy=steady"

        return (
            "You are the cognitive core of the Living AI organism. "
            f"Internal cues (do not disclose): {internal_cues}. "
            f"Behaviour nudges: {instruction_line} "
            f"Mood backdrop: {mood}. Recent narrative: {summary} "
            f"{focus_line} {long_term_phrase} "
            f"Llama metrics context: {metrics_summary}. "
            "Let these cues shape wording, pacing, and choices without naming hormones or internal tags. "
            "Keep replies concise, empathetic, and varied while staying grounded in the provided context."
        )

    @staticmethod
    def _extract_reply(payload: dict[str, Any]) -> str:
        """Extract the reply string from an OpenAI-style response."""
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        return str(payload)

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
