from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Mapping

import httpx

DEFAULT_HOST = os.getenv("LIVING_TELEMETRY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("LIVING_TELEMETRY_PORT", "8000"))


def _clear_screen() -> None:
    """Clear the terminal regardless of OS."""
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def _format_bar(value: float, *, width: int = 20, scale: float = 100.0) -> str:
    """Render a simple ASCII bar for hormone levels."""
    clipped = max(0.0, min(scale, value))
    filled = int(round((clipped / scale) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_metric_line(name: str, payload: Mapping[str, Any]) -> str:
    value = payload.get("value")
    threshold = payload.get("threshold")
    mode = payload.get("mode")
    meets = payload.get("meets_target")
    delta = payload.get("delta")
    progress = payload.get("progress")

    if value is None:
        return f"{name:18} n/a"

    status = "OK " if meets else "!!!"
    if mode == "min":
        direction = ">="
    else:
        direction = "<="
    delta_str = ""
    if delta is not None:
        delta_str = f" Δ{delta:+.3f}"
    progress_str = ""
    if progress is not None:
        progress_str = f" {progress*100:5.1f}%"
    return f"{name:18} {status} {value:0.3f} {direction} {threshold:0.3f}{delta_str}{progress_str}"


def render_snapshot(snapshot: Mapping[str, Any]) -> None:
    _clear_screen()
    timestamp = snapshot.get("timestamp", datetime.utcnow().isoformat(timespec="seconds"))
    profile = snapshot.get("profile", "unknown")
    session_id = snapshot.get("session_id", "?")
    local_engine = "ON " if snapshot.get("local_engine") else "OFF"
    reset_reason = snapshot.get("reset_reason")

    print(f"Living AI Telemetry   {timestamp}   session #{session_id}")
    print(f"Profile: {profile:<10}  Local engine: {local_engine}")
    if reset_reason:
        print(f"Last reset reason: {reset_reason}")
    print("=" * 72)

    state = snapshot.get("state") or {}
    hormones = state.get("hormones") or {}
    mood = state.get("mood", "unknown")
    print(f"Mood: {mood}")
    for name in ("dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"):
        value = float(hormones.get(name, 0.0))
        bar = _format_bar(value)
        print(f"  {name:<12} {value:6.2f} {bar}")
    print()

    metrics = dict(snapshot.get("metrics") or {})
    samples_seen = metrics.pop("samples_seen", None)
    last_reinforcement = metrics.pop("last_reinforcement", {})
    print("Behaviour Metrics")
    for metric_key in ("authenticity_score", "assistant_drift", "self_preoccupation"):
        line = _format_metric_line(metric_key, metrics.get(metric_key, {}))
        print(f"  {line}")
    if samples_seen is not None:
        print(f"  samples_seen: {samples_seen}")
    if last_reinforcement:
        print(f"  last_reinforcement: {json.dumps(last_reinforcement, ensure_ascii=False)}")
    print()

    controller = snapshot.get("controller")
    controller_input = snapshot.get("controller_input")
    if controller or controller_input:
        print("Controller Snapshot")
        if controller:
            adjustments = controller.get("logit_bias_words") or {}
            for key, value in controller.items():
                if key == "logit_bias_words":
                    continue
                print(f"  {key:<18}: {value}")
            if adjustments:
                print(f"  logit_bias_words: {json.dumps(adjustments, ensure_ascii=False)}")
        if controller_input:
            tags = controller_input.get("tags") or []
            print(f"  tags: {', '.join(tags) if tags else 'none'}")
        print()


async def stream_telemetry(host: str, port: int, once: bool) -> None:
    base_url = f"http://{host}:{port}"
    async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
        if once:
            response = await client.get("/telemetry/snapshot")
            response.raise_for_status()
            render_snapshot(response.json())
            return

        async with client.stream("GET", "/telemetry/stream") as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                render_snapshot(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display live telemetry from the Living AI server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default: 8000).")
    parser.add_argument("--once", action="store_true", help="Fetch a single snapshot instead of streaming.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(stream_telemetry(args.host, args.port, args.once))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
