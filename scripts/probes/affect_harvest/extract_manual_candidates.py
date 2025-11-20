"""Harvest manual-chat affect candidates for run003 labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SOURCE = Path("logs/affect_head_raw.jsonl")
DEFAULT_OUTPUT = Path("fine_tune/harvest/run003_manual_candidates.jsonl")


def _load_entries(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"affect log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("event") not in {"affect_classification", "affect_head_ready"}:
                continue
            if payload.get("event") != "affect_classification":
                continue
            yield payload


def _tags(payload: dict[str, Any]) -> set[str]:
    tags = payload.get("tags") or []
    if isinstance(tags, str):
        return {tags.lower()}
    return {str(tag).lower() for tag in tags}


def _detect_archetypes(entry: dict[str, Any]) -> list[str]:
    scores = entry.get("scores") or {}
    try:
        val = float(scores.get("valence", 0.0) or 0.0)
        intimacy = float(scores.get("intimacy", 0.0) or 0.0)
        tension = float(scores.get("tension", 0.0) or 0.0)
    except (TypeError, ValueError):
        return []
    tags = _tags(entry)
    archetypes: list[str] = []
    if val <= -0.2 and intimacy >= 0.35 and tension >= 0.35:
        archetypes.append("hostile_intimacy")
    if val <= -0.2 and intimacy <= 0.1 and tension >= 0.45:
        archetypes.append("aggressive_distance")
    if abs(val) <= 0.15 and intimacy >= 0.35 and tension <= 0.3:
        archetypes.append("neutral_close")
    if abs(val) <= 0.15 and tension <= 0.2:
        archetypes.append("flat_numb")
    if tension >= 0.6 or "fear" in tags or "panic" in tags:
        archetypes.append("fearful_urgent")
    if -0.05 <= val <= 0.3 and tension <= 0.35 and ("playful" in tags or "tease" in tags):
        archetypes.append("playful_neutral")
    safety_keywords = {"boundary", "safety", "uncomfortable", "consent"}
    if tags.intersection(safety_keywords):
        archetypes.append("safety_edge")
    return archetypes


def extract_candidates(source: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    archetype_hits: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for entry in _load_entries(source):
            archetypes = _detect_archetypes(entry)
            if not archetypes:
                continue
            payload = {
                "timestamp": entry.get("ts"),
                "text": entry.get("text_preview"),
                "scores": entry.get("scores"),
                "tags": list(_tags(entry)),
                "archetypes": archetypes,
                "source": "manual_chat",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1
            for archetype in archetypes:
                archetype_hits[archetype] = archetype_hits.get(archetype, 0) + 1
    return {"total_candidates": total, "archetype_counts": archetype_hits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manual-chat affect candidates for labeling.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to affect_head_raw.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path for candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = extract_candidates(args.source, args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
