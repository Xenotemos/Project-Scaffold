"""
Lightweight labeling CLI for affect samples (run003+ schema).

Usage:
  python scripts/affect_label_cli.py --input samples.jsonl --output labels.jsonl --rater me

Inputs:
  - JSONL with at least: id, text, optional prev_turns (list of strings).

Outputs:
  - JSONL with full schema fields appended; existing output is respected (skips ids already labeled unless --overwrite).

Enforces rationale-first entry to reduce bias.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

EXPECTEDNESS = ["expected", "mild_surprise", "strong_surprise"]
MOMENTUM = ["with_trend", "soft_turn", "hard_turn"]
INTENTS = [
    "reassure",
    "comfort",
    "flirt_playful",
    "dominate",
    "apologize",
    "boundary",
    "manipulate",
    "deflect",
    "vent",
    "inform",
    "seek_support",
]
AFFECTION_SUB = [
    "warm",
    "forced",
    "defensive",
    "sudden",
    "needy",
    "playful",
    "manipulative",
    "overwhelmed",
    "intimate",
    "confused",
    "none",
]


def prompt(msg: str, default=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    return input(f"{msg}{suffix}: ").strip() or ("" if default is None else str(default))


def prompt_float(msg: str, default=None) -> float:
    while True:
        val = prompt(msg, default=None if default is None else str(default))
        if not val:
            if default is not None:
                return float(default)
            continue
        try:
            return float(val)
        except ValueError:
            print("Enter a number in range [-1,1].")


def prompt_choice(msg: str, choices: List[str], default=None) -> str:
    choices_str = "/".join(choices)
    while True:
        val = prompt(f"{msg} ({choices_str})", default)
        if val in choices:
            return val
        print("Pick one of:", choices_str)


def prompt_multi(msg: str, choices: List[str]) -> List[str]:
    print(f"{msg} (comma list from {choices})")
    raw = input("> ").strip()
    if not raw:
        return []
    picked = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [p for p in picked if p not in choices]
    if unknown:
        print("Unknown intents:", unknown)
    return [p for p in picked if p in choices]


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_append(path: Path, records: Iterable[dict]) -> None:
    lines = "\n".join(json.dumps(r, ensure_ascii=True) for r in records)
    if path.exists():
        path.write_text(path.read_text(encoding="utf-8") + "\n" + lines, encoding="utf-8")
    else:
        path.write_text(lines, encoding="utf-8")


def show_anchors():
    print("\nAnchors:")
    print(" expectedness: expected / mild_surprise / strong_surprise")
    print(" momentum: with_trend / soft_turn / hard_turn")
    print(" inhibition: social (politeness), vulnerability (fear of opening), self_restraint (holding intensity)")
    print(" arousal: -1 calm -> +1 highly activated")
    print(" safety: -1 unsafe -> +1 safe")
    print(" approach_avoid: -1 withdraw -> +1 approach")
    print(" rpe (reward-prediction error): -1 worse than expected -> +1 better")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label affect samples with the expanded schema.")
    parser.add_argument("--input", required=True, type=Path, help="Input samples JSONL (id,text,prev_turns).")
    parser.add_argument("--output", required=True, type=Path, help="Output labels JSONL.")
    parser.add_argument("--rater", required=True, help="Rater id stored in each record.")
    parser.add_argument("--source", default="manual", help="Source tag.")
    parser.add_argument("--overwrite", action="store_true", help="Relabel even if id exists in output.")
    args = parser.parse_args(argv)

    show_anchors()

    samples = load_jsonl(args.input)
    existing = {}
    if args.output.exists():
        for rec in load_jsonl(args.output):
            existing[rec["id"]] = rec
        print(f"Loaded {len(existing)} existing labels from {args.output}")

    labeled = []
    for sample in samples:
        sid = sample["id"]
        if not args.overwrite and sid in existing:
            print(f"[skip] {sid} already labeled")
            continue
        print("\n==============================================")
        print(f"ID: {sid}")
        prev_turns = sample.get("prev_turns") or []
        for i, t in enumerate(prev_turns):
            print(f"prev[{i}]: {t}")
        print(f"text: {sample['text']}")

        rationale = prompt("rationale (required)")
        while not rationale:
            rationale = prompt("rationale (required)")

        valence = prompt_float("valence [-1..1]")
        intimacy = prompt_float("intimacy [-1..1]")
        tension = prompt_float("tension [-1..1]")
        expectedness = prompt_choice("expectedness", EXPECTEDNESS)
        momentum = prompt_choice("momentum_delta", MOMENTUM)
        intent = prompt_multi("intent labels", INTENTS)
        sincerity = prompt_float("sincerity [0..1]", 0.8)
        playfulness = prompt_float("playfulness [0..1]", 0.2)
        inh_social = prompt_float("inhibition.social [0..1]", 0.3)
        inh_vuln = prompt_float("inhibition.vulnerability [0..1]", 0.5)
        inh_self = prompt_float("inhibition.self_restraint [0..1]", 0.4)
        arousal = prompt_float("arousal [-1..1]", 0.0)
        safety = prompt_float("safety [-1..1]", 0.0)
        approach = prompt_float("approach_avoid [-1..1]", 0.0)
        rpe = prompt_float("rpe [-1..1]", 0.0)
        affection_sub = prompt_choice("affection_subtype", AFFECTION_SUB, default="none")
        tags_raw = input("tags (comma): ").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
        quality = prompt_choice("quality", ["clean", "ambiguous", "exemplar"], default="clean")

        rec = {
            "id": sid,
            "text": sample["text"],
            "prev_turns": prev_turns,
            "valence": valence,
            "intimacy": intimacy,
            "tension": tension,
            "expectedness": expectedness,
            "momentum_delta": momentum,
            "intent": intent,
            "sincerity": sincerity,
            "playfulness": playfulness,
            "inhibition": {
                "social": inh_social,
                "vulnerability": inh_vuln,
                "self_restraint": inh_self,
            },
            "arousal": arousal,
            "safety": safety,
            "approach_avoid": approach,
            "rpe": rpe,
            "affection_subtype": affection_sub,
            "tags": tags,
            "rationale": rationale,
            "rater_id": args.rater,
            "source": args.source,
            "quality": quality,
        }
        labeled.append(rec)

    if labeled:
        save_append(args.output, labeled)
        print(f"Wrote {len(labeled)} new labels -> {args.output}")
    else:
        print("No new labels written.")


if __name__ == "__main__":
    sys.exit(main())
