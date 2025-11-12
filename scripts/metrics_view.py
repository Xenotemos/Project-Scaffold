#!/usr/bin/env python3
"""Render simple ASCII trends for reinforcement metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

BARS = " .:-=+*#%@"
DEFAULT_FILE = Path(__file__).resolve().parents[1] / "logs" / "reinforcement_metrics.jsonl"


def _load_metrics(path: Path, limit: int | None) -> list[dict[str, float]]:
    if not path.exists():
        return []
    entries: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                entries.append({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
    if limit is not None and limit > 0:
        return entries[-limit:]
    return entries


def _sparkline(values: Sequence[float]) -> str:
    if not values:
        return "(no data)"
    low = min(values)
    high = max(values)
    span = high - low
    if math.isclose(span, 0.0, abs_tol=1e-6):
        index = min(len(BARS) - 1, max(0, int(len(BARS) / 2)))
        return BARS[index] * len(values)
    scaled = []
    for value in values:
        ratio = (value - low) / span
        index = min(len(BARS) - 1, max(0, int(ratio * (len(BARS) - 1))))
        scaled.append(BARS[index])
    return "".join(scaled)


def _summary(values: Iterable[float]) -> str:
    values = list(values)
    if not values:
        return "avg 0.000 | min 0.000 | max 0.000"
    avg = sum(values) / len(values)
    return f"avg {avg:.3f} | min {min(values):.3f} | max {max(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ASCII sparklines for reinforcement metrics.")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help="Path to reinforcement_metrics.jsonl (defaults to project logs directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Number of recent samples to include (0 means all).",
    )
    args = parser.parse_args()

    metrics = _load_metrics(args.file, args.limit if args.limit > 0 else None)
    if not metrics:
        print("No reinforcement metrics found.")
        return

    keys = ("authenticity_score", "assistant_drift", "self_preoccupation")
    print(f"Loaded {len(metrics)} samples from {args.file}")
    for key in keys:
        series = [entry.get(key, 0.0) for entry in metrics]
        print(f"{key:>20}: {_sparkline(series)}")
        print(f"{'':>21}{_summary(series)}")


if __name__ == "__main__":
    main()
