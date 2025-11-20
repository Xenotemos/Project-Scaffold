"""Select a balanced subset from external affect candidates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("fine_tune/harvest/run003_external_candidates.jsonl")
DEFAULT_OUTPUT = Path("fine_tune/harvest/run003_external_subset.jsonl")

TARGET_COUNTS = {
    "hostile_intimacy": 700,
    "aggressive_distance": 700,
    "neutral_close": 600,
    "flat_numb": 500,
    "fearful_urgent": 500,
    "playful_neutral": 400,
    "safety_edge": 300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced external candidates for run003 labeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to full external candidates JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL for sampled subset.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility.")
    parser.add_argument("--counts", type=str, default="", help="Optional JSON mapping archetype->count to override defaults.")
    return parser.parse_args()


def load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def select_subset(records: list[dict[str, Any]], counts: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    bucketed: dict[str, list[dict[str, Any]]] = {}
    for entry in records:
        archetype = entry.get("archetype")
        if archetype not in counts:
            continue
        bucketed.setdefault(archetype, []).append(entry)
    random.seed(seed)
    selection: dict[str, list[dict[str, Any]]] = {}
    for archetype, target in counts.items():
        bucket = bucketed.get(archetype, [])
        if not bucket:
            continue
        if len(bucket) <= target:
            selection[archetype] = bucket
            continue
        selection[archetype] = random.sample(bucket, target)
    return selection


def write_subset(selection: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for archetype, entries in selection.items():
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                label_row = {
                    "archetype": archetype,
                    "text": entry.get("text"),
                    "source": entry.get("source"),
                    "suggested_scores": entry.get("scores"),
                }
                label_path = output.with_suffix(".label.jsonl")
                with label_path.open("a", encoding="utf-8") as label_handle:
                    label_handle.write(json.dumps(label_row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    counts = dict(TARGET_COUNTS)
    if args.counts:
        try:
            counts.update(json.loads(args.counts))
        except json.JSONDecodeError:
            pass
    records = load_candidates(args.input)
    selection = select_subset(records, counts, args.seed)
    write_subset(selection, args.output)
    summary = {archetype: len(entries) for archetype, entries in selection.items()}
    summary["total"] = sum(summary.values())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
