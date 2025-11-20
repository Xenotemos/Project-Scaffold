"""Extract affect archetype candidates from external corpora (GoEmotions, EmpatheticDialogues)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_GOEMOTIONS_INPUT = Path("data/goemotions.jsonl")
DEFAULT_EMPATHETIC_INPUT = Path("data/empathetic_dialogues.csv")
DEFAULT_OUTPUT = Path("fine_tune/harvest/run003_external_candidates.jsonl")

GOEMOTIONS_MAP: dict[str, dict[str, Any]] = {
    "anger": {"archetype": "aggressive_distance", "valence": -0.75, "intimacy": -0.15, "tension": 0.9},
    "annoyance": {"archetype": "aggressive_distance", "valence": -0.55, "intimacy": -0.1, "tension": 0.7},
    "disgust": {"archetype": "aggressive_distance", "valence": -0.8, "intimacy": -0.3, "tension": 0.85},
    "fear": {"archetype": "fearful_urgent", "valence": -0.35, "intimacy": 0.2, "tension": 0.95},
    "apprehension": {"archetype": "fearful_urgent", "valence": -0.2, "intimacy": 0.1, "tension": 0.8},
    "caring": {"archetype": "neutral_close", "valence": 0.1, "intimacy": 0.55, "tension": 0.25},
    "desire": {"archetype": "hostile_intimacy", "valence": 0.2, "intimacy": 0.65, "tension": 0.5},
    "jealousy": {"archetype": "hostile_intimacy", "valence": -0.5, "intimacy": 0.6, "tension": 0.85},
    "grief": {"archetype": "flat_numb", "valence": -0.6, "intimacy": 0.3, "tension": 0.65},
    "sadness": {"archetype": "flat_numb", "valence": -0.4, "intimacy": 0.2, "tension": 0.4},
    "remorse": {"archetype": "safety_edge", "valence": -0.35, "intimacy": 0.35, "tension": 0.6},
    "trust": {"archetype": "neutral_close", "valence": 0.15, "intimacy": 0.6, "tension": 0.2},
    "relief": {"archetype": "neutral_close", "valence": 0.3, "intimacy": 0.4, "tension": 0.25},
    "boredom": {"archetype": "flat_numb", "valence": -0.05, "intimacy": 0.0, "tension": 0.1},
    "excitement": {"archetype": "playful_neutral", "valence": 0.35, "intimacy": 0.45, "tension": 0.4},
}

EMPATHETIC_MAP: dict[str, dict[str, Any]] = {
    "jealous": GOEMOTIONS_MAP["jealousy"],
    "furious": GOEMOTIONS_MAP["anger"],
    "afraid": GOEMOTIONS_MAP["fear"],
    "cautious": GOEMOTIONS_MAP["apprehension"],
    "lonely": {"archetype": "neutral_close", "valence": -0.15, "intimacy": 0.45, "tension": 0.35},
    "content": {"archetype": "neutral_close", "valence": 0.2, "intimacy": 0.4, "tension": 0.2},
    "numb": GOEMOTIONS_MAP["boredom"],
    "devastated": GOEMOTIONS_MAP["grief"],
    "confident": {"archetype": "playful_neutral", "valence": 0.25, "intimacy": 0.35, "tension": 0.3},
}


def _load_goemotions(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
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
                    yield payload
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
        return []
    if path.is_dir():
        candidates = list(path.glob("*.jsonl")) + list(path.glob("*.csv"))
        for candidate in candidates:
            yield from _load_goemotions(candidate)
    return []


def _load_empathetic_dialogues(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_dir():
        for file in path.glob("*.csv"):
            yield from _load_empathetic_dialogues(file)
        return []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def extract_goemotions(input_path: Path, writer) -> int:
    count = 0
    for entry in _load_goemotions(input_path):
        text = entry.get("text")
        if not isinstance(text, str):
            continue
        normalized: list[str] = []
        labels_field = entry.get("labels")
        if isinstance(labels_field, list):
            normalized.extend(str(label).lower() for label in labels_field)
        else:
            for label, meta in GOEMOTIONS_MAP.items():
                value = entry.get(label)
                try:
                    if float(value) >= 0.5:
                        normalized.append(label)
                except (TypeError, ValueError):
                    continue
        if not normalized and "emotion" in entry:
            normalized.append(str(entry["emotion"]).lower())
        for label in normalized:
            meta = GOEMOTIONS_MAP.get(label)
            if not meta:
                continue
            payload = {
                "source": "goemotions",
                "label": label,
                "archetype": meta["archetype"],
                "text": text.strip(),
                "scores": {
                    "valence": meta["valence"],
                    "intimacy": meta["intimacy"],
                    "tension": meta["tension"],
                },
            }
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def extract_empathetic(input_path: Path, writer) -> int:
    count = 0
    for row in _load_empathetic_dialogues(input_path):
        utterance = row.get("utterance") or row.get("utterance_text")
        emotion = row.get("emotion") or row.get("emotion_label") or row.get("context")
        if not isinstance(utterance, str) or not isinstance(emotion, str):
            continue
        label = emotion.lower().strip()
        meta = EMPATHETIC_MAP.get(label)
        if not meta:
            continue
        payload = {
            "source": "empathetic_dialogues",
            "label": label,
            "archetype": meta["archetype"],
            "text": utterance.strip(),
            "scores": {
                "valence": meta["valence"],
                "intimacy": meta["intimacy"],
                "tension": meta["tension"],
            },
        }
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract external affect candidates.")
    parser.add_argument("--goemotions", type=Path, default=DEFAULT_GOEMOTIONS_INPUT, help="Path to GoEmotions JSONL/CSV or directory.")
    parser.add_argument("--empathetic", type=Path, default=DEFAULT_EMPATHETIC_INPUT, help="Path to EmpatheticDialogues CSV or directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_go = 0
    total_ed = 0
    with args.output.open("w", encoding="utf-8") as handle:
        if args.goemotions.exists():
            total_go = extract_goemotions(args.goemotions, handle)
        if args.empathetic.exists():
            total_ed = extract_empathetic(args.empathetic, handle)
    summary = {"goemotions": total_go, "empathetic_dialogues": total_ed, "total": total_go + total_ed}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
