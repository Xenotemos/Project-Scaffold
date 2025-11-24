import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bin_counts(values: Sequence[float], edges: Sequence[float]) -> Counter:
    counts = Counter()
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
    return counts


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_file(path: Path, edges: List[float], weight_field: str | None) -> None:
    data = load_jsonl(path)
    axes = ("valence", "intimacy", "tension")
    vals = {ax: [d[ax] for d in data] for ax in axes}

    print(f"\n{path}")
    print(f"rows: {len(data)}")

    for ax in axes:
        arr = vals[ax]
        print(
            f"{ax:8} min {min(arr):.3f} max {max(arr):.3f} mean {np.mean(arr):.3f} "
            f"std {np.std(arr):.3f}"
        )
        bc = bin_counts(arr, edges)
        print(f"  bins {dict(sorted(bc.items()))}")
        top = Counter(arr).most_common(5)
        print(f"  top5 {top}")

    print(
        f"corr val-int {corr(vals['valence'], vals['intimacy']):.3f} "
        f"val-ten {corr(vals['valence'], vals['tension']):.3f} "
        f"int-ten {corr(vals['intimacy'], vals['tension']):.3f}"
    )

    if weight_field and all(weight_field in d for d in data):
        weights = [d[weight_field] for d in data]
        print(
            f"weights min {np.min(weights):.4f} max {np.max(weights):.4f} "
            f"mean {np.mean(weights):.4f}"
        )


def parse_edges(arg: str) -> List[float]:
    parts = [float(x) for x in arg.split(",")]
    if sorted(parts) != parts:
        raise ValueError("edges must be sorted ascending")
    return parts


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize affect dataset JSONL files (valence/intimacy/tension)."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL file(s) to summarize")
    parser.add_argument(
        "--edges",
        type=parse_edges,
        default="-1.0,-0.7,-0.3,0.3,0.7,1.0",
        help="Comma-separated bin edges (inclusive lower, exclusive upper).",
    )
    parser.add_argument(
        "--weight-field",
        default="sample_weight",
        help="Optional field name for weights; set to '' to skip.",
    )
    args = parser.parse_args(argv)

    edges = args.edges
    if isinstance(edges, str):
        edges = parse_edges(edges)

    weight_field = args.weight_field or None
    for path in args.paths:
        summarize_file(path, edges, weight_field)


if __name__ == "__main__":
    main()
