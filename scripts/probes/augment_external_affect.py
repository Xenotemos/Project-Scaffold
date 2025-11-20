"""Convert locally stored GoEmotions CSV into affect JSONL with V/I/T targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

# V/I/T mapping for GoEmotions labels; extend if needed.
EMO_MAP: Dict[str, Tuple[float, float, float]] = {
    "admiration": (0.6, 0.5, 0.25),
    "amusement": (0.65, 0.45, 0.25),
    "anger": (-0.7, 0.2, 0.9),
    "annoyance": (-0.4, 0.2, 0.6),
    "approval": (0.5, 0.4, 0.25),
    "caring": (0.65, 0.7, 0.35),
    "confusion": (-0.1, 0.2, 0.55),
    "curiosity": (0.2, 0.25, 0.45),
    "desire": (0.7, 0.7, 0.35),
    "disappointment": (-0.5, 0.3, 0.55),
    "disapproval": (-0.5, 0.25, 0.6),
    "disgust": (-0.7, 0.2, 0.85),
    "embarrassment": (-0.4, 0.35, 0.6),
    "excitement": (0.8, 0.6, 0.4),
    "fear": (-0.5, 0.3, 0.85),
    "gratitude": (0.75, 0.55, 0.2),
    "grief": (-0.65, 0.7, 0.7),
    "joy": (0.8, 0.6, 0.2),
    "love": (0.8, 0.7, 0.25),
    "nervousness": (-0.4, 0.3, 0.75),
    "optimism": (0.45, 0.4, 0.25),
    "pride": (0.55, 0.4, 0.35),
    "realization": (0.2, 0.25, 0.3),
    "relief": (0.4, 0.5, 0.25),
    "remorse": (-0.55, 0.35, 0.65),
    "sadness": (-0.6, 0.6, 0.6),
    "surprise": (0.2, 0.2, 0.6),
    "neutral": (0.0, 0.1, 0.1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GoEmotions CSV to affect JSONL.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(r"C:\Users\USER\Desktop\LLM_training\goemotions\goemotions.csv"),
        help="Path to the local GoEmotions CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fine_tune/affect_dataset_ext.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=5,
        help="Minimum text length to keep.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12000,
        help="Limit number of rows (0 = no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_written = 0
    with args.csv.open("r", encoding="utf-8") as f_in, args.output.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        labels = [c for c in reader.fieldnames or [] if c in EMO_MAP]
        for row in reader:
            if args.max_rows > 0 and rows_written >= args.max_rows:
                break
            text = (row.get("text") or "").strip()
            if len(text) < args.min_len:
                continue
            # Pick first positive label
            label = None
            for lab in labels:
                try:
                    if float(row.get(lab, "0") or 0) > 0:
                        label = lab
                        break
                except ValueError:
                    continue
            if not label:
                continue
            vit = EMO_MAP[label]
            payload = {
                "text": text,
                "valence": vit[0],
                "intimacy": vit[1],
                "tension": vit[2],
                "tags": [label],
                "stimulus": "goemotions_csv",
            }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            rows_written += 1
    print(f"Wrote {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
