from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _run_step(label: str, command: Iterable[str], env: Mapping[str, str] | None = None) -> None:
    command = list(command)
    print(f"[ci] {label}: {' '.join(command)}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(command, cwd=str(ROOT), env=merged_env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 1e-9 or var_b <= 1e-9:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _enforce_affect_dataset_metrics(
    data_path: Path, corr_threshold: float = 0.25, confusion_cap: float = 0.25
) -> None:
    samples = _load_jsonl(data_path)
    if not samples:
        print(f"[ci] affect dataset missing or empty at {data_path} (skipping decorrelation check)", file=sys.stderr)
        return
    val = []
    intim = []
    tense = []
    safety = []
    for row in samples:
        try:
            val.append(float(row.get("valence", 0.0)))
            intim.append(float(row.get("intimacy", 0.0)))
            tense.append(float(row.get("tension", 0.0)))
            safety.append(float(row.get("safety", 0.0)))
        except Exception:
            continue
    c_val_int = _corr(val, intim)
    c_val_ten = _corr(val, tense)
    print(f"[ci] affect decorrelation: corr(val,int)={c_val_int:+.3f} corr(val,ten)={c_val_ten:+.3f}")
    if abs(c_val_int) > corr_threshold:
        raise SystemExit(f"[ci] affect decorrelation failed: |corr(val,int)|={abs(c_val_int):.3f} > {corr_threshold}")
    # charge/safety confusion: safety strongly negative but intimacy high
    unsafe_intimate = 0
    total = 0
    for s, i in zip(safety, intim):
        total += 1
        if s < -0.2 and i > 0.45:
            unsafe_intimate += 1
    if total:
        confusion = unsafe_intimate / total
        print(f"[ci] affect safety/intimacy confusion={confusion:.3f} (cap {confusion_cap:.3f})")
        if confusion > confusion_cap:
            raise SystemExit(
                f"[ci] affect safety/intimacy confusion too high: {confusion:.3f} > {confusion_cap:.3f}"
            )


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
    parser.add_argument(
        "--affect-dev-data",
        default="docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.jsonl",
        help="Path to dev/eval JSONL for affect decorrelation checks.",
    )
    parser.add_argument(
        "--affect-corr-threshold",
        type=float,
        default=0.25,
        help="Maximum allowed absolute corr(val,int) on the dev set (default: 0.25).",
    )
    parser.add_argument(
        "--affect-confusion-cap",
        type=float,
        default=0.25,
        help="Maximum fraction of unsafe-but-intimate rows (safety<-0.2 & intimacy>0.45).",
    )
    parser.add_argument(
        "--disable-local-llama",
        action="store_true",
        default=True,
        help="Disable local llama engine during CI steps to avoid heavy startup (default: enabled).",
    )
    parser.add_argument(
        "--disable-affect-sidecar",
        action="store_true",
        default=True,
        help="Disable affect sidecar during CI steps (default: enabled).",
    )
    args = parser.parse_args()

    env_overrides: dict[str, str] = {}
    if args.disable_local_llama:
        env_overrides["LLAMA_DISABLE"] = "1"
    if args.disable_affect_sidecar:
        env_overrides["AFFECT_SIDECAR_DISABLE"] = "1"

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
        _run_step("HORMONE_MODEL_DRY_RUN", train_base, env=env_overrides)

        train_controller = [
            sys.executable,
            "scripts/train_controller_policy.py",
            "--log-file",
            str(log_path),
            "--dry-run",
        ]
        _run_step("CONTROLLER_POLICY_DRY_RUN", train_controller, env=env_overrides)

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
        _run_step("PROBE_CANARY", probe_cmd, env=env_overrides)

    if not args.skip_affect:
        affect_cmd = [
            sys.executable,
            "-m",
            "scripts.probes.affect_harvest.affect_validation",
            "--profiles",
            *args.profiles,
            "--log-path",
            args.log_file,
            "--json-out",
            args.affect_json,
        ]
        if args.disable_local_llama or args.disable_affect_sidecar:
            affect_cmd.append("--soft-fail")
        _run_step("AFFECT_VALIDATION", affect_cmd, env=env_overrides)
        if args.disable_local_llama or args.disable_affect_sidecar:
            print("[ci] affect expectations present but skipped because local llama/affect sidecar are disabled.")
        else:
            _enforce_affect_threshold(args.affect_json, threshold=args.affect_threshold)
        data_path = ROOT / args.affect_dev_data
        _enforce_affect_dataset_metrics(
            data_path,
            corr_threshold=args.affect_corr_threshold,
            confusion_cap=args.affect_confusion_cap,
        )

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
