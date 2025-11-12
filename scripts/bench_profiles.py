from __future__ import annotations

import argparse
import asyncio
import os
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, Tuple

from scripts import bench_chat

PROFILE_FILES: Dict[str, str] = {
    "instruct": "settings.json",
    "base": "settings.base.json",
}


def _swap_settings_file(file_name: str) -> Tuple[str | None, bool]:
    previous = os.environ.get("LIVING_SETTINGS_FILE")
    os.environ["LIVING_SETTINGS_FILE"] = file_name
    return previous, True


def _restore_settings_file(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("LIVING_SETTINGS_FILE", None)
    else:
        os.environ["LIVING_SETTINGS_FILE"] = previous


def _summarise(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    lengths = [row.get("length", 0) for row in rows]
    sources = Counter(row.get("source", "unknown") for row in rows)
    heuristic_turns = sum(1 for row in rows if row.get("source") != "local")
    return {
        "turns": len(rows),
        "avg_length": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "max_length": max(lengths) if lengths else 0,
        "heuristic_turns": heuristic_turns,
        "source_counts": dict(sources),
    }


async def _run_profile(profile: str, settings_file: str) -> Dict[str, Any]:
    prev, _ = _swap_settings_file(settings_file)
    try:
        rows = await bench_chat.run_bench()
    finally:
        _restore_settings_file(prev)
    return {
        "profile": profile,
        "settings_file": settings_file,
        "summary": _summarise(rows),
        "rows": rows,
    }


async def main_async(profiles: Iterable[str]) -> None:
    results = []
    for profile in profiles:
        if profile not in PROFILE_FILES:
            raise SystemExit(f"Unknown profile '{profile}'. Known profiles: {', '.join(PROFILE_FILES)}")
        settings = PROFILE_FILES[profile]
        print(f"=== Running bench for profile '{profile}' ({settings}) ===")
        result = await _run_profile(profile, settings)
        results.append(result)
        summary = result["summary"]
        print("Summary:", summary)
        print()
    print("=== Combined Overview ===")
    for result in results:
        name = result["profile"]
        summary = result["summary"]
        print(f"{name}: turns={summary['turns']} avg_len={summary['avg_length']} max_len={summary['max_length']} "
              f"heuristic_turns={summary['heuristic_turns']} sources={summary['source_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench multiple model profiles via llama.cpp.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILE_FILES.keys()),
        help="Profiles to bench (default: instruct base).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.profiles))


if __name__ == "__main__":
    main()
