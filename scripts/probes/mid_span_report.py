#!/usr/bin/env python3
"""Summarise mid-span harness results with quick ASCII trends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

BARS = " .:-=+*#%@"
DEFAULT_BASE = Path("logs") / "probe_runs" / "mid_span"


def _sparkline(values: Sequence[float]) -> str:
    if not values:
        return "(no data)"
    low = min(values)
    high = max(values)
    span = high - low
    if math.isclose(span, 0.0, abs_tol=1e-6):
        index = min(len(BARS) - 1, max(0, int(len(BARS) / 2)))
        return BARS[index] * len(values)
    scaled = []
    for value in values:
        ratio = (value - low) / span if span else 0.0
        index = min(len(BARS) - 1, max(0, int(ratio * (len(BARS) - 1))))
        scaled.append(BARS[index])
    return "".join(scaled)


def _summary(values: Iterable[float]) -> str:
    series = list(values)
    if not series:
        return "avg 0.000 | min 0.000 | max 0.000"
    avg = sum(series) / len(series)
    return f"avg {avg:.3f} | min {min(series):.3f} | max {max(series):.3f}"


def _load_summary(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list payload.")
    return data


def _latest_harness(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [entry for entry in base.iterdir() if entry.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _resolve_summary(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.summary:
        summary_path = args.summary
        return summary_path.parent, summary_path
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_harness(args.base_dir)
        if run_dir is None:
            raise SystemExit(f"No harness directories found under {args.base_dir}.")
    summary_path = run_dir / "summary_compact.json"
    if not summary_path.exists():
        raise SystemExit(f"summary_compact.json not found in {run_dir}")
    return run_dir, summary_path


def _load_meta(run_dir: Path) -> dict[str, object] | None:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Render quick trend reports for mid-span harness runs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE,
        help="Root directory containing harness-* runs (default: logs/probe_runs/mid_span).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific harness directory to inspect (overrides --base-dir lookup).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Path to a summary_compact.json file (overrides --run-dir).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="Show the last N iterations per profile in the detail table (default: 10).",
    )
    parser.add_argument(
        "--require-streak",
        action="store_true",
        help="Exit with code 1 if any profile lacks a >=2-turn low-self authenticity streak.",
    )
    args = parser.parse_args()

    run_dir, summary_path = _resolve_summary(args)
    rows = _load_summary(summary_path)
    if not rows:
        raise SystemExit(f"{summary_path} is empty.")
    rows.sort(key=lambda item: (item.get("iteration", 0), item.get("profile", "")))
    meta = _load_meta(run_dir)

    print(f"Run directory: {run_dir}")
    if meta:
        requested = meta.get("requested_minutes")
        elapsed = meta.get("elapsed_seconds")
        iterations = meta.get("iterations_completed")
        print(f"  Requested minutes : {requested}")
        print(f"  Elapsed seconds   : {elapsed:.1f}" if isinstance(elapsed, (int, float)) else f"  Elapsed seconds   : {elapsed}")
        print(f"  Iterations        : {iterations}")
    print(f"Summary source: {summary_path}")

    profiles: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        profile = str(row.get("profile") or "unknown")
        profiles.setdefault(profile, []).append(row)

    best_streaks: dict[str, int] = {}
    for profile, series in sorted(profiles.items()):
        streak = 0
        best_streak = 0
        streak_iters: list[tuple[int, float, float]] = []
        best_streak_iters: list[tuple[int, float, float]] = []
        for item in series:
            auth_ok = (item.get("gate") or {}).get("authenticity_score", {}).get("ok", False)
            self_ok = (item.get("self_preoccupation") is not None) and float(item.get("self_preoccupation", 1.0)) <= 0.6
            if auth_ok and self_ok:
                streak += 1
                streak_iters.append(
                    (
                        int(item.get("iteration", 0)),
                        float(item.get("authenticity", 0.0) or 0.0),
                        float(item.get("self_preoccupation", 0.0) or 0.0),
                    )
                )
                if streak > best_streak:
                    best_streak = streak
                    best_streak_iters = streak_iters.copy()
            else:
                streak = 0
                streak_iters.clear()
        best_streaks[profile] = best_streak
        iterations = [int(item.get("iteration", 0)) for item in series]
        auth = [float(item.get("authenticity", 0.0) or 0.0) for item in series]
        self_vals = [float(item.get("self_preoccupation", 0.0) or 0.0) for item in series]
        outward = [float(item.get("outward_streak", 0.0) or 0.0) for item in series]
        gate_hits = [
            item.get("iteration")
            for item in series
            if (item.get("gate") or {}).get("authenticity_score", {}).get("ok")
            and (item.get("gate") or {}).get("self_preoccupation", {}).get("ok")
        ]
        print()
        print(f"Profile: {profile} ({len(series)} iterations)")
        print(f"  authenticity     : {_sparkline(auth)}")
        print(f"                    { _summary(auth) }")
        print(f"  self_preoccupation: {_sparkline(self_vals)}")
        print(f"                    { _summary(self_vals) }")
        print(f"  outward_streak   : {_sparkline(outward)}")
        print(f"                    { _summary(outward) }")
        print(f"  Gate hits        : {gate_hits or 'none'}")
        if best_streak >= 2:
            streak_str = ", ".join(f"{iter_id} (auth={auth:.3f}, self={self_val:.3f})" for iter_id, auth, self_val in best_streak_iters)
            print(f"  Longest low-self streak ({best_streak}): {streak_str}")
        else:
            print("  Longest low-self streak: none (need >=2 consecutive)")
        tail = series[-args.last :] if args.last > 0 else series
        if tail:
            print("  Recent iterations:")
            print("    iter | auth  | self  | outward")
            for item in tail:
                print(
                    f"    {int(item.get('iteration', 0)):>4} | "
                    f"{float(item.get('authenticity', 0.0) or 0.0):>0.3f} | "
                    f"{float(item.get('self_preoccupation', 0.0) or 0.0):>0.3f} | "
                    f"{float(item.get('outward_streak', 0.0) or 0.0):>0.3f}"
                )
    if args.require_streak:
        missing = [profile for profile, streak in best_streaks.items() if streak < 2]
        if missing:
            print()
            print(f"Streak requirement not met for: {', '.join(missing)}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
