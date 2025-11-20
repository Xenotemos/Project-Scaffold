"""Export cleaned affect probe rows into the training dataset JSONL."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_INPUT = Path("logs/probe_runs/affect_data.jsonl")
DEFAULT_OUTPUT = Path("fine_tune/affect_dataset.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter probe output into a compact affect_dataset.jsonl for LoRA training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Probe JSONL input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Training JSONL output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--min-len", type=int, default=8, help="Minimum prompt length (default 8).")
    parser.add_argument("--max-len", type=int, default=260, help="Maximum prompt length (default 260).")
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Optional profile whitelist (e.g., instruct base). If omitted, keep all.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle outputs before writing.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate prompts (case-insensitive) to reduce repetition.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _project(row: Mapping[str, object]) -> dict[str, object]:
    targets = row.get("targets") or {}
    return {
        "text": row.get("text", ""),
        "valence": targets.get("valence", 0.0),
        "intimacy": targets.get("intimacy", 0.0),
        "tension": targets.get("tension", 0.0),
        "tags": row.get("tags") or [],
        "stimulus": row.get("stimulus"),
    }


def main() -> None:
    args = parse_args()
    rows = []
    profiles = set(args.profiles) if args.profiles else None
    seen_text: set[str] = set()
    for row in _load_rows(args.input):
        raw_text = str(row.get("text", "")).strip()
        text = unicodedata.normalize("NFKC", raw_text)
        text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
        if len(text) < args.min_len or len(text) > args.max_len:
            continue
        profile = row.get("profile")
        if profiles and profile not in profiles:
            continue
        if args.dedupe:
            key = text.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
        projected = _project(dict(row, text=text))
        rows.append(projected)

    if args.shuffle:
        from random import shuffle

        shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
