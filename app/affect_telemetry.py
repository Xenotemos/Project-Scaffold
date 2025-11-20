"""Lightweight telemetry logger for the affect head sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def log_affect_head_event(log_path: Path, event: Mapping[str, Any]) -> None:
    """Append a compact JSON line to the affect head telemetry log."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Telemetry is best-effort; don't raise.
        return


__all__ = ["log_affect_head_event"]
