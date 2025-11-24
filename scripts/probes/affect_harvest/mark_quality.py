import json
from pathlib import Path

"""
Heuristic quality marker for labeled JSONL (schema) to set quality and optional weight boost.
Rules:
 - safety/threat/boundary/high-tension + clear intent -> exemplar
 - clear playful/affection subtypes with consistent tone -> clean/exemplar
 - otherwise if non-empty rationale -> clean
 - else ambiguous
Weights (optional):
 - exemplar: *3.0
 - clean:    *1.5
 - ambiguous:*1.0

Usage:
  python scripts/probes/affect_harvest/mark_quality.py --input file.jsonl --output file_marked.jsonl [--set-weights]
"""

import argparse
import re


def load(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--set-weights", action="store_true", help="apply exemplar/clean/ambiguous multipliers to sample_weight")
    args = p.parse_args()

    data = load(args.input)

    threat_rx = re.compile(r"(threat|unsafe|danger|creepy|kill|stab|back off|touch me|step back|stop)", re.I)
    playful_rx = re.compile(r"(;\)|;\-|lol|lmao|haha|heartthrob|superstar|cute|gorgeous|handsome|wink)", re.I)
    affection_rx = re.compile(r"(love you|proud of you|miss you)", re.I)

    for d in data:
        text = d.get("text", "")
        intents = set(d.get("intent", []))
        aff = d.get("affection_subtype", "none")
        val = d.get("valence", 0)
        ten = d.get("tension", 0)
        safety = d.get("safety", 0)
        rationale = d.get("rationale", "").strip()

        quality = "ambiguous"
        # Exemplars: clear safety/boundary or fear/urgent with tension & negative safety
        if ("boundary" in intents or threat_rx.search(text)) and ten >= 0.6 and safety <= 0:
            quality = "exemplar"
        elif aff in {"warm", "playful", "intimate", "defensive", "manipulative", "needy"} and rationale:
            quality = "clean"
        elif playful_rx.search(text) and aff == "playful":
            quality = "clean"
        elif affection_rx.search(text) and aff in {"warm", "intimate"} and val > 0:
            quality = "clean"
        elif rationale:
            quality = "clean"

        d["quality"] = quality

        if args.set_weights:
            w = d.get("sample_weight", 1.0)
            if quality == "exemplar":
                w *= 3.0
            elif quality == "clean":
                w *= 1.5
            d["sample_weight"] = w

    save(args.output, data)
    print(f"wrote {len(data)} rows -> {args.output}")


if __name__ == "__main__":
    main()
