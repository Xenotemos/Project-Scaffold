"""Compact console monitor for the Concurrent Affect Head Module (CAHM).

Shows two sections:
- Top: current input preview, affect scores, tags, latency, and a terse
  emotional/hormonal interpretation.
- Bottom: the model's reasoning/processing text.

Unlike a plain tail, this clears and redraws to keep the view compact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from pathlib import Path
from typing import Any, Iterable


def _clear_screen() -> None:
    try:
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
    except Exception:
        # If clearing fails, continue; telemetry should still be readable.
        pass


def _affect_label(valence: float, intimacy: float, tension: float) -> str:
    tone = "neutral"
    if valence > 0.25:
        tone = "positive"
    elif valence < -0.25:
        tone = "negative"
    arousal = "tense" if tension > 0.35 else "calm"
    closeness = "connected" if intimacy > 0.25 else ("distant" if intimacy < -0.25 else "neutral")
    return f"{tone}, {arousal}, {closeness}"


def _hormone_hint(valence: float, intimacy: float, tension: float) -> str:
    hints: list[str] = []
    if valence > 0.2:
        hints.append("serotonin↑")
        hints.append("oxytocin↑")
        hints.append("cortisol↓")
    elif valence < -0.2:
        hints.append("cortisol↑")
        hints.append("dopamine↓")
    if intimacy > 0.2:
        hints.append("bonding↑")
    if tension > 0.35:
        hints.append("alert↑")
    return ", ".join(hints) if hints else "steady"


def _render(event: dict[str, Any]) -> None:
    _clear_screen()
    ts = event.get("ts") or event.get("timestamp") or ""
    source = event.get("source", "affect_head")
    text = event.get("text_preview") or event.get("text") or ""
    scores = event.get("scores") or {}
    val = float(scores.get("valence", 0.0) or 0.0)
    intimacy = float(scores.get("intimacy", 0.0) or 0.0)
    tension = float(scores.get("tension", 0.0) or 0.0)
    conf = float(scores.get("confidence", 0.0) or 0.0)
    latency = event.get("latency_ms")
    tags = event.get("tags") or []
    reasoning = (event.get("reasoning") or "").strip()

    affect_read = _affect_label(val, intimacy, tension)
    hormone_read = _hormone_hint(val, intimacy, tension)

    print(f"CAHM Telemetry   {ts}   src={source}")
    print("=" * 72)
    print("Input / Affect")
    print(f"  user: {text}")
    if tags:
        print(f"  tags: {', '.join(tags)}")
    score_line = (
        f"  affect: val {val:+.2f} | in {intimacy:+.2f} | te {tension:+.2f} | conf {conf:.2f}"
    )
    print(score_line)
    print(f"  affect read: {affect_read}")
    if latency is not None:
        print(f"  latency: {latency} ms")
    print(f"  hormone hint: {hormone_read}")
    print()
    print("Reasoning / Processing")
    if reasoning:
        wrapped = textwrap.wrap(reasoning, width=72)
        for line in wrapped:
            print(f"  {line}")
    else:
        print("  (no reasoning text captured)")

    sys.stdout.flush()


def _render_placeholder(message: str) -> None:
    _clear_screen()
    print("CAHM Telemetry")
    print("=" * 72)
    print(message)
    sys.stdout.flush()


def _line_stream(path: Path, poll_seconds: float = 0.5) -> Iterable[str]:
    """Yield new lines appended to a file, handling rotation."""
    position = 0
    while True:
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(position)
                while True:
                    line = handle.readline()
                    if line:
                        position = handle.tell()
                        yield line.rstrip("\n")
                        continue
                    time.sleep(poll_seconds)
                    try:
                        current_size = path.stat().st_size
                    except FileNotFoundError:
                        position = 0
                        break
                    if current_size < position:
                        position = 0
                        break
        except FileNotFoundError:
            time.sleep(poll_seconds)
            position = 0
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(poll_seconds)



def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor CAHM telemetry in a compact view.")
    parser.add_argument("--log", default="logs/affect_head_raw.jsonl", help="Path to the JSONL telemetry log.")
    args = parser.parse_args()
    log_path = Path(args.log)

    _render_placeholder(f"Waiting for CAHM events in {log_path} ...")
    last_event: dict[str, Any] | None = None
    for line in _line_stream(log_path):
        try:
            event = json.loads(line)
            last_event = event
            _render(event)
        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            break
        except Exception:
            # Keep the monitor alive on unexpected parsing errors.
            _render_placeholder("Monitor recovered from a read error; waiting for next event...")
            time.sleep(0.5)

    if last_event:
        _render(last_event)
    else:
        _render_placeholder("Monitor stopped without receiving events.")


if __name__ == "__main__":
    main()
