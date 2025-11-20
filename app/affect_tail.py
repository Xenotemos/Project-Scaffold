"""Readable tail log for affect head classifications."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def append_affect_raw(base_path: Path, payload: Mapping[str, Any]) -> None:
    """Append the full raw payload from the affect head."""
    path = base_path / "affect_head_raw.jsonl"
    _append_jsonl(path, payload)


def append_affect_compact(base_path: Path, entry: Mapping[str, Any]) -> None:
    """Append a compact human-friendly line to affect_head_readable.log."""
    path = base_path / "affect_head_readable.log"
    ts = entry.get("ts") or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    text = entry.get("text_preview") or ""
    scores = entry.get("scores") or {}
    tags = entry.get("tags") or []
    latency = entry.get("latency_ms")
    source = entry.get("source") or "affect_head"
    reasoning = entry.get("reasoning") or ""
    line = (
        f"[{ts}] src={source} v={scores.get('valence', 0):+.2f} "
        f"in={scores.get('intimacy', 0):+.2f} te={scores.get('tension', 0):+.2f} "
        f"conf={scores.get('confidence', 0):.2f} tags={tags} lat_ms={latency} :: {text}"
    )
    reason_line = f"    reason: {reasoning}" if reasoning else ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            if reason_line:
                handle.write(reason_line + "\n")
    except Exception:
        return


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


__all__ = ["append_affect_raw", "append_affect_compact"]
