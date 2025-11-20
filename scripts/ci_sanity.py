from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def _run_step(label: str, command: Iterable[str]) -> None:
    command = list(command)
    print(f"[ci] {label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CI sanity checks for hormone/controller training and probes.")
    parser.add_argument(
        "--log-file",
        default="logs/endocrine_turns.jsonl",
        help="Endocrine turn log used for dry-run retraining (default: logs/endocrine_turns.jsonl).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["instruct", "base"],
        help="Profiles passed to the probe canary (default: instruct base).",
    )
    parser.add_argument(
        "--turn",
        action="append",
        help="Override probe turns (may be provided multiple times).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip dry-run retraining steps.",
    )
    parser.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip the single-iteration probe check.",
    )
    parser.add_argument(
        "--skip-affect",
        action="store_true",
        help="Skip the affect validation probe.",
    )
    parser.add_argument(
        "--probe-log",
        default="logs/probe_runs/ci_probe.jsonl",
        help="Destination for the CI probe log (default: logs/probe_runs/ci_probe.jsonl).",
    )
    parser.add_argument(
        "--mid-span-dir",
        type=Path,
        help="Optional mid-span harness directory to validate for consecutive low-self streaks.",
    )
    parser.add_argument(
        "--affect-json",
        default="logs/probe_runs/affect_validation_ci.json",
        help="Destination for affect validation JSON summary (default: logs/probe_runs/affect_validation_ci.json).",
    )
    parser.add_argument(
        "--affect-threshold",
        type=float,
        default=0.0,
        help="Minimum fraction of expectations that must pass (default: 0.0, i.e., require all expectations).",
    )
    args = parser.parse_args()

    log_path = ROOT / args.log_file
    if not log_path.exists() and not args.skip_train:
        print(f"[ci] endocrine log missing at {log_path}", file=sys.stderr)
        raise SystemExit(2)

    if not args.skip_train:
        train_base = [
            sys.executable,
            "scripts/train_hormone_model.py",
            "--log-file",
            str(log_path),
            "--dry-run",
        ]
        _run_step("HORMONE_MODEL_DRY_RUN", train_base)

        train_controller = [
            sys.executable,
            "scripts/train_controller_policy.py",
            "--log-file",
            str(log_path),
            "--dry-run",
        ]
        _run_step("CONTROLLER_POLICY_DRY_RUN", train_controller)

    if not args.skip_probes:
        probe_cmd = [
            sys.executable,
            "-m",
            "scripts.continuous_probes",
            "--profiles",
            *args.profiles,
            "--iterations",
            "1",
            "--no-retain-rows",
            "--reset-session",
            "--idle-seconds",
            "0",
            "--log-file",
            args.probe_log,
        ]
        if args.turn:
            for turn in args.turn:
                probe_cmd.extend(["--turn", turn])
        else:
            probe_cmd.extend(
                [
                    "--turn",
                    "before anything else, tell me what you notice in your body right now.",
                ]
            )
        _run_step("PROBE_CANARY", probe_cmd)

    if not args.skip_affect:
        affect_cmd = [
            sys.executable,
            "-m",
            "scripts.probes.affect_validation",
            "--profiles",
            *args.profiles,
            "--log-path",
            args.log_file,
            "--json-out",
            args.affect_json,
        ]
        _run_step("AFFECT_VALIDATION", affect_cmd)
        _enforce_affect_threshold(args.affect_json, threshold=args.affect_threshold)

    if args.mid_span_dir:
        run_dir = args.mid_span_dir
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        if not run_dir.exists():
            print(f"[ci] mid-span directory missing at {run_dir}", file=sys.stderr)
            raise SystemExit(3)
        streak_cmd = [
            sys.executable,
            "-m",
            "scripts.probes.mid_span_report",
            "--run-dir",
            str(run_dir),
            "--require-streak",
        ]
        _run_step("MID_SPAN_STREAK_GUARD", streak_cmd)

    print("[ci] sanity checks completed successfully.")


if __name__ == "__main__":
    main()


def _enforce_affect_threshold(json_path: str | Path, threshold: float = 0.0) -> None:
    path = Path(json_path)
    if not path.exists():
        print(f"[ci] affect validation summary missing at {path}", file=sys.stderr)
        raise SystemExit(4)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    summaries = data.get("summaries") or []
    unmet = [
        result
        for result in results
        if any(not spec.get("met") for spec in (result.get("expectation") or {}).values())
    ]
    total_expectations = sum(summary.get("expectation_total", 0) for summary in summaries)
    total_hits = sum(summary.get("expectation_hits", 0) for summary in summaries)
    fraction = 1.0 if total_expectations == 0 else total_hits / total_expectations
    if unmet or fraction < threshold:
        print(
            f"[ci] affect validation failed: hits={total_hits}/{total_expectations} fraction={fraction:.3f} threshold={threshold:.3f}",
            file=sys.stderr,
        )
        for record in unmet[:5]:
            scenario = record.get("scenario", {}).get("name")
            profile = record.get("profile")
            expectations = record.get("expectation") or {}
            failed = [k for k, spec in expectations.items() if not spec.get("met")]
            print(
                f"[ci]  - profile={profile} scenario={scenario} failed={','.join(failed)}",
                file=sys.stderr,
            )
        raise SystemExit(5)
