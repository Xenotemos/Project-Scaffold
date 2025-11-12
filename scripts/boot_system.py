"""Bootstrap the Living AI stack (llama-server + FastAPI UI) in one command.

This script mirrors the environment setup from `llama_env.ps1/sh`, ensures the
local llama.cpp server is running, starts the FastAPI app via uvicorn, and opens
the browser UI when ready.

It is designed to be PyInstaller-friendly so it can be packaged as an `.exe`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable

import httpx

try:
    import msvcrt  # type: ignore[attr-defined]
except ImportError:
    msvcrt = None  # type: ignore[assignment]


def _detect_root() -> Path:
    """Locate the project root whether running as script or PyInstaller bundle."""
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(getattr(sys, "_MEIPASS")))
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.cwd().resolve())

    for base in candidates:
        for candidate in [base, *base.parents]:
            if (candidate / "main.py").exists() and (candidate / "scripts").exists():
                return candidate
    raise FileNotFoundError("Unable to locate project root (expected to find main.py).")


ROOT = _detect_root()
DEFAULT_LLAMA_BIN = ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
DEFAULT_MODEL = Path(r"D:\AI\LLMs\mistral-7b-instruct-v0.2.Q4_K_M.gguf")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _detect_vulkan() -> None:
    """Mimic the PowerShell helper by prepending the latest Vulkan SDK bin to PATH."""
    if os.getenv("VULKAN_SDK"):
        return
    default_root = Path("C:/VulkanSDK")
    if not default_root.exists():
        return
    candidates = sorted(default_root.glob("*"), reverse=True)
    for directory in candidates:
        bin_path = directory / "Bin"
        if bin_path.exists():
            os.environ["VULKAN_SDK"] = str(directory)
            os.environ["PATH"] = f"{bin_path}{os.pathsep}{os.environ.get('PATH','')}"
            break


def _ensure_env(args: argparse.Namespace) -> dict[str, str]:
    """Populate the environment variables consumed by main.py / LocalLlamaEngine."""
    env = os.environ.copy()
    llama_bin = Path(args.llama_bin or env.get("LLAMA_SERVER_BIN") or DEFAULT_LLAMA_BIN)
    model_path = Path(args.model or env.get("LLAMA_MODEL_PATH") or DEFAULT_MODEL)

    if not llama_bin.exists():
        raise FileNotFoundError(f"llama-server binary not found at {llama_bin}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    env.setdefault("LLAMA_SERVER_BIN", str(llama_bin))
    env.setdefault("LLAMA_MODEL_PATH", str(model_path))
    env.setdefault("LLAMA_SERVER_HOST", args.llama_host or env.get("LLAMA_SERVER_HOST", DEFAULT_HOST))
    env.setdefault("LLAMA_SERVER_PORT", str(args.llama_port or env.get("LLAMA_SERVER_PORT", "8080")))
    env.setdefault("LLAMA_MODEL_ALIAS", env.get("LLAMA_MODEL_ALIAS", "mistral-local"))
    extra_args = env.get("LLAMA_SERVER_ARGS", "--ctx-size 4096 --no-webui")
    if args.llama_args:
        extra_args = args.llama_args
    env["LLAMA_SERVER_ARGS"] = extra_args
    env.setdefault("LLAMA_SERVER_READY_TIMEOUT", str(args.llama_ready_timeout or 120))
    env.setdefault("LLAMA_SERVER_TIMEOUT", str(args.llama_timeout or 60))
    env.setdefault("LIVING_LLM_TIMEOUT", str(max(int(env["LLAMA_SERVER_TIMEOUT"]), 60)))
    return env


async def _await_ping(host: str, port: int, timeout: float = 60.0) -> None:
    """Poll the FastAPI `/ping` endpoint until it responds."""
    url = f"http://{host}:{port}/ping"
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=10.0)) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
            except Exception:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for FastAPI to respond at {url}")


async def _warmup_chat(host: str, port: int, message: str, timeout: float = 60.0) -> dict[str, Any]:
    """Dispatch a warmup chat request to trigger the local llama engine."""
    url = f"http://{host}:{port}/chat"
    payload = {"message": message, "stimulus": "reward"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=timeout)) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _register_console_close_handler(get_proc: "Callable[[], subprocess.Popen[Any] | None]") -> "Callable[[], None]":
    """Ensure child uvicorn is terminated when the console window closes."""
    if os.name != "nt":
        return lambda: None

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    def _terminate_child() -> None:
        proc = get_proc()
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def handler(ctrl_type: int) -> int:
        if ctrl_type in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT):
            _terminate_child()
        return 0  # pass to next handler

    c_handler = HandlerRoutine(handler)
    installed = kernel32.SetConsoleCtrlHandler(c_handler, True)

    def unregister() -> None:
        if installed:
            kernel32.SetConsoleCtrlHandler(c_handler, False)

    return unregister


async def _await_attach_exit(proc: subprocess.Popen[Any], args: argparse.Namespace) -> None:
    """Block until uvicorn exits, auto-exit timers trigger, or ENTER is pressed."""
    loop = asyncio.get_running_loop()

    wait_tasks: dict[asyncio.Future[Any], str] = {}
    proc_wait = loop.run_in_executor(None, proc.wait)
    wait_tasks[proc_wait] = "proc"

    if args.auto_exit > 0:
        auto_task = asyncio.create_task(asyncio.sleep(args.auto_exit))
        wait_tasks[auto_task] = "auto"

    enter_task: asyncio.Task[Any] | None = None
    if not args.no_enter_stop:
        prompt = "[boot] Press ENTER to stop uvicorn...\n"
        enter_task = asyncio.create_task(asyncio.to_thread(input, prompt))
        wait_tasks[enter_task] = "enter"
    else:
        if args.auto_exit <= 0:
            print("[boot] Attach mode enabled. Use Ctrl+C or terminate the process to stop uvicorn.")

    print("[boot] Attach mode enabled.")
    if enter_task and args.auto_exit <= 0:
        print("[boot] Waiting for ENTER (or Ctrl+C) to stop uvicorn.")
    elif enter_task and args.auto_exit > 0:
        print(f"[boot] Waiting for ENTER or auto-exit after {args.auto_exit} seconds.")
    elif args.auto_exit > 0:
        print(f"[boot] Auto-exit scheduled after {args.auto_exit} seconds.")

    done, pending = await asyncio.wait(wait_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
    winner = next(iter(done))
    reason = wait_tasks[winner]

    for fut in pending:
        fut.cancel()

    if reason == "proc":
        try:
            code = winner.result()
        except Exception:
            code = None
        print(f"[boot] uvicorn exited on its own{'' if code is None else f' (code {code})'}.")
        return
    if reason == "auto":
        print(f"[boot] Auto-exit timer elapsed ({args.auto_exit} seconds).")
        return
    print("[boot] ENTER received; stopping uvicorn.")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Boot llama.cpp and FastAPI UI together.")
    parser.add_argument("--llama-bin", help="Path to llama-server executable.")
    parser.add_argument("--model", help="Path to GGUF model file.")
    parser.add_argument("--llama-host", help="Host for llama-server (defaults to 127.0.0.1).")
    parser.add_argument("--llama-port", type=int, help="Port for llama-server (defaults to 8080).")
    parser.add_argument("--llama-args", help="Override LLAMA_SERVER_ARGS.")
    parser.add_argument("--llama-ready-timeout", type=int, help="Seconds to wait for llama-server readiness.")
    parser.add_argument("--llama-timeout", type=int, help="Seconds for llama-server request timeout.")
    parser.add_argument("--ui-host", default=DEFAULT_HOST, help="FastAPI host (default 127.0.0.1).")
    parser.add_argument("--ui-port", type=int, default=DEFAULT_PORT, help="FastAPI port (default 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the initial /chat warmup request.")
    parser.add_argument("--warmup-message", default="System warmup check.", help="Message used for warmup chat.")
    parser.add_argument(
        "--stay-attached",
        action="store_true",
        help="Wait for uvicorn to exit instead of detaching after startup.",
    )
    parser.add_argument(
        "--auto-exit",
        type=int,
        default=0,
        help="Automatically stop uvicorn after N seconds (attach mode only).",
    )
    parser.add_argument(
        "--no-enter-stop",
        action="store_true",
        help="Do not wait for ENTER key in attach mode.",
    )
    args = parser.parse_args(argv)

    _detect_vulkan()
    env = _ensure_env(args)

    uvicorn_path = ROOT / ".venv-win" / "Scripts" / "uvicorn.exe"
    if not uvicorn_path.exists():
        raise FileNotFoundError(f"uvicorn executable not found at {uvicorn_path}")

    uvicorn_cmd = [
        str(uvicorn_path),
        "main:app",
        "--log-level",
        "info",
        "--host",
        args.ui_host,
        "--port",
        str(args.ui_port),
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(uvicorn_cmd, cwd=ROOT, env=env, creationflags=creationflags)
    proc_holder: dict[str, subprocess.Popen[Any] | None] = {"proc": proc}
    deregister_close = _register_console_close_handler(lambda: proc_holder["proc"])

    def _handle_sigint(signum: int, _frame: Any) -> None:
        if proc.poll() is None:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    detach = not args.stay_attached

    try:
        await _await_ping(args.ui_host, args.ui_port, timeout=120.0)
        print(f"[boot] FastAPI UI ready at http://{args.ui_host}:{args.ui_port}/")
        warmup_summary = None
        if not args.no_warmup:
            warmup_summary = await _warmup_chat(args.ui_host, args.ui_port, args.warmup_message)
            print("[boot] Warmup chat response:")
            print(json.dumps(warmup_summary, indent=2))
        if not args.no_browser:
            webbrowser.open(f"http://{args.ui_host}:{args.ui_port}/")
        if detach:
            print(f"[boot] Detaching; uvicorn running at PID {proc.pid}.")
            proc_holder["proc"] = None
            proc = None  # skip cleanup so server keeps running
            return 0
        await _await_attach_exit(proc, args)
        return proc.returncode or 0
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            proc_holder["proc"] = None
            deregister_close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
