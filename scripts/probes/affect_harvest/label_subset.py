"""Interactive labeling CLI for affect subset entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

DEFAULT_INPUT = Path("fine_tune/harvest/run003_external_subset.label.jsonl")
DEFAULT_OUTPUT = Path("fine_tune/harvest/run003_external_labeled.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label affect subset entries interactively.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Label JSONL produced by select_external_subset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL for approved/labeled rows.")
    parser.add_argument("--resume-id", type=int, default=0, help="Line number to resume from (0-based).")
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"label file not found: {path}")
    rows: list[dict[str, Any]] = []
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
                rows.append(payload)
    return rows


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
    while True:
        raw = input(f"{prompt} [{default if default is not None else ''}] > ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if -1.0 <= value <= 1.0:
                return value
            print("Value must be between -1.0 and 1.0.")
        except ValueError:
            print("Invalid number; try again.")


def _prompt_tags(default_tags: list[str] | None = None) -> list[str]:
    default = ", ".join(default_tags or [])
    raw = input(f"Tags (comma-separated) [{default}] > ").strip()
    if not raw:
        return default_tags or []
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def label_entries(entries: list[dict[str, Any]], resume: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for idx, row in enumerate(entries):
            if idx < resume:
                continue
            text = row.get("text", "").strip()
            archetype = row.get("archetype")
            suggested = row.get("suggested_scores") or row.get("scores") or {}
            print("=" * 72)
            print(f"Entry #{idx+1}/{len(entries)}  archetype={archetype}")
            print(f"Source: {row.get('source')}")
            print(f"Text: {text}")
            print(f"Suggested: val={suggested.get('valence')}  intimacy={suggested.get('intimacy')}  tension={suggested.get('tension')}")
            keep = input("Keep this entry? [y/N] > ").strip().lower()
            if keep not in {"y", "yes"}:
                continue
            valence = _prompt_float("Valence", float(suggested.get("valence", 0.0)))
            intimacy = _prompt_float("Intimacy", float(suggested.get("intimacy", 0.0)))
            tension = _prompt_float("Tension", float(suggested.get("tension", 0.0)))
            confidence = _prompt_float("Confidence (0-1)", 0.6)
            tags = _prompt_tags([archetype] if archetype else [])
            payload = {
                "text": text,
                "valence": round(valence or 0.0, 4),
                "intimacy": round(intimacy or 0.0, 4),
                "tension": round(tension or 0.0, 4),
                "confidence": round(confidence or 0.0, 4) if confidence is not None else None,
                "tags": tags,
                "source": row.get("source"),
                "archetype": archetype,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print("Saved.")


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input)
    label_entries(entries, args.resume_id, args.output)


if __name__ == "__main__":
    main()
