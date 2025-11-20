"""Lifecycle manager for the dedicated affect llama.cpp sidecar."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore[unused-ignore]
except Exception as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

DEFAULT_LOG_PATH = Path("logs/affect_llama.log")


class AffectSidecarManager:
    """Start, monitor, and stop the affect llama.cpp server."""

    def __init__(self, config: Mapping[str, Any], log_path: Path | None = None) -> None:
        self.config = dict(config)
        self.log_path = log_path or DEFAULT_LOG_PATH
        self._process: subprocess.Popen[str] | None = None
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def base_url(self) -> str:
        host = str(self.config.get("host", "127.0.0.1"))
        port = int(self.config.get("port", 8082))
        return f"http://{host}:{port}"

    async def ensure_running(self) -> None:
        """Start the sidecar if it is not already running; wait until healthy."""
        async with self._lock:
            if await self._healthy():
                return
            await self._start()
            await self._wait_ready()

    async def stop(self) -> None:
        async with self._lock:
            if self._client:
                await self._client.aclose()
                self._client = None
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except Exception:
                    self._process.kill()
            self._process = None

    # Convenience for synchronous callers (e.g., during startup from a threadpool)
    def ensure_running_sync(self) -> None:
        asyncio.run(self.ensure_running())

    async def _start(self) -> None:
        binary = Path(self.config.get("llama_server_bin") or "")
        model = Path(self.config.get("model_path") or "")
        if not binary.exists():
            raise FileNotFoundError(f"affect llama-server binary not found at {binary}")
        if not model.exists():
            raise FileNotFoundError(f"affect model not found at {model}")
        host = str(self.config.get("host", "127.0.0.1"))
        port = int(self.config.get("port", 8082))
        alias = str(self.config.get("alias", "affect-head"))
        extra_args = self.config.get("extra_args") or []
        if isinstance(extra_args, str):
            extra_args = shlex.split(extra_args)
        cmd = [
            str(binary),
            "--model",
            str(model),
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            alias,
        ]
        cmd.extend(map(str, extra_args))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = self.log_path.open("a", encoding="utf-8")
        stderr = subprocess.STDOUT
        creation_flags = 0
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        self._process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )

    async def _wait_ready(self, timeout: float | None = None) -> None:
        if HTTPX_IMPORT_ERROR:
            raise RuntimeError(f"httpx required for readiness probe: {HTTPX_IMPORT_ERROR}")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        deadline = asyncio.get_event_loop().time() + (timeout or float(self.config.get("readiness_timeout", 15.0)))
        url = f"{self.base_url}/v1/models"
        while True:
            if self._process and self._process.poll() not in (None, 0):
                raise RuntimeError("affect llama.cpp exited during startup")
            try:
                resp = await self._client.get(url, timeout=5.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(f"Timed out waiting for affect llama.cpp at {url}")
            await asyncio.sleep(0.25)

    async def _healthy(self) -> bool:
        if HTTPX_IMPORT_ERROR:
            return False
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False


__all__ = ["AffectSidecarManager"]
