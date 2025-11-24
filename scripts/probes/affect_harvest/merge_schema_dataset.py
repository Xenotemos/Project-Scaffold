import json
from pathlib import Path
from typing import List

"""
Merge reviewed schema-labeled splits plus heuristic backfill into a single training file,
optionally excluding dev/gold IDs, and produce train/dev splits.

Usage:
  python scripts/probes/affect_harvest/merge_schema_dataset.py \
      --inputs fine_tune/harvest/affect_run003_schema_backfill.jsonl docs/planning/CAHM rework/affect_gold_labels.jsonl \
      --exclude_ids docs/planning/CAHM rework/affect_gold_labels.jsonl \
      --train fine_tune/affect_run003_schema_train.jsonl \
      --dev   fine_tune/affect_run003_schema_dev.jsonl \
      --dev_ratio 0.1

Notes:
  - Excludes any record whose id matches exclude set.
  - If sample_weight missing, assigns 1.0.
"""


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_jsonl(path: Path, rows: List[dict]):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows), encoding="utf-8")


def main():
    import argparse
    import random

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--exclude_ids", nargs="*", default=[], type=Path)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--dev", required=True, type=Path)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    args = p.parse_args()

    exclude = set()
    for path in args.exclude_ids:
        for row in load_jsonl(path):
            if "id" in row and row["id"] is not None:
                exclude.add(str(row["id"]))

    rows = []
    for path in args.inputs:
        for row in load_jsonl(path):
            rid = row.get("id")
            if rid is not None and str(rid) in exclude:
                continue
            if "sample_weight" not in row:
                row["sample_weight"] = 1.0
            rows.append(row)

    random.seed(42)
    random.shuffle(rows)
    dev_size = int(len(rows) * args.dev_ratio)
    dev_rows = rows[:dev_size]
    train_rows = rows[dev_size:]

    args.train.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.train, train_rows)
    save_jsonl(args.dev, dev_rows)
    print(f"merged {len(rows)} rows -> train {len(train_rows)} dev {len(dev_rows)}")


if __name__ == "__main__":
    main()
