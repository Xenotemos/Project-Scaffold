"""Sample slices of affect_dataset.jsonl for quick label verification."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping

DEFAULT_PATH = Path("fine_tune/affect_dataset.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print random samples per scenario/tags from affect_dataset.jsonl for label auditing."
    )
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH, help="Dataset JSONL path.")
    parser.add_argument(
        "--per-scenario", type=int, default=5, help="Number of samples to show per stimulus/tag cluster."
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42, help="Seed for deterministic sampling."
    )
    return parser.parse_args()


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    args = parse_args()
    random.seed(args.shuffle_seed)
    buckets = defaultdict(list)
    for row in load_rows(args.path):
        key = (row.get("stimulus") or "unknown", tuple(row.get("tags") or []))
        buckets[key].append(row)

    for (stimulus, tags), rows in sorted(buckets.items()):
        print(f"\n=== stimulus={stimulus} tags={list(tags)} ({len(rows)} rows) ===")
        sample = rows if len(rows) <= args.per_scenario else random.sample(rows, args.per_scenario)
        for idx, row in enumerate(sample, 1):
            print(f"[{idx}] text: {row.get('text')}")
            print(
                f"    targets: V={row.get('valence')} I={row.get('intimacy')} T={row.get('tension')}"
            )


if __name__ == "__main__":
    main()
