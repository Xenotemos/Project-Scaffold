from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
import traceback

import main as app_main
from scripts import continuous_probes as cp

DEFAULT_DURATION_MINUTES = 45.0
DEFAULT_INTERACTIONS = 5
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_LOG_DIR = Path(cp.LOG_DIR_DEFAULT) / "mid_span"


def _apply_auto_tweak(profile: str, iteration: int, log_path: Path | str) -> None:
    """Extend clamp settings mid-run when the gate hasn't been met."""
    recovery_floor = 6
    priming_floor = 5
    bias_spike = getattr(app_main, "RESET_PRIMING_BIAS_DEFAULT", 0.35)
    app_main.CLAMP_RECOVERY_TURNS = max(app_main.CLAMP_RECOVERY_TURNS, recovery_floor)
    app_main.CLAMP_PRIMING_TURNS = max(app_main.CLAMP_PRIMING_TURNS, priming_floor)
    app_main.RESET_PRIMING_BIAS = bias_spike
    log_path = Path(log_path)
    payload = {
        "timestamp": cp._now_iso(),
        "event": "auto_tweak",
        "profile": profile,
        "iteration_triggered": iteration,
        "actions": {
            "clamp_recovery_floor": recovery_floor,
            "priming_floor": priming_floor,
            "bias_spike": round(bias_spike, 3),
        },
    }
    try:
        cp._log_probe_result(log_path, payload)
    except Exception as exc:
        print(
            f"[mid_span_probes][auto] failed to log tweak event for {profile}: {exc}"
        )
    print(
        f"[mid_span_probes][auto] profile={profile} applied longer recovery/priming window "
        f"after failing both gates for {iteration} iterations."
    )


async def _cooldown_sleep(total_seconds: float) -> None:
    if total_seconds <= 0:
        return
    start = datetime.now().astimezone()
    target = start + timedelta(seconds=total_seconds)
    target_iso = target.isoformat(timespec="seconds")
    print(
        f"[mid_span_probes] Cooldown {total_seconds:.1f}s before next iteration "
        f"(next at {target_iso})"
    )
    remaining = total_seconds
    update_interval = 1.0 if total_seconds <= 30 else 5.0
    while remaining > 0:
        sleep_step = min(update_interval, remaining)
        await asyncio.sleep(sleep_step)
        remaining = max(0.0, (target - datetime.now().astimezone()).total_seconds())
        if remaining > 0:
            if remaining >= 60:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                display = f"{minutes}m {seconds:02d}s"
            else:
                display = f"{remaining:.0f}s"
            print(f"[mid_span_probes]   cooldown remaining: {display}")
    print("[mid_span_probes] Cooldown complete; starting next iteration.")


def _resolve_turns(args: argparse.Namespace) -> Sequence[str]:
    if args.turn:
        turns = tuple(args.turn)
        return turns
    if args.interactions <= 0:
        raise SystemExit("--interactions must be at least 1 when --turn is not supplied.")
    if args.interactions > len(cp.DEFAULT_TURNS):
        raise SystemExit(
            f"--interactions={args.interactions} exceeds the available default turns "
            f"({len(cp.DEFAULT_TURNS)}). Supply explicit --turn prompts instead."
        )
    return cp.DEFAULT_TURNS[: args.interactions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configurable mid-span probe session (approx. 30-60 minutes) that injects a handful "
            "of scripted interactions and logs behavioural and endocrine metrics to JSONL."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(cp.PROFILE_FILES.keys()),
        help="Profiles to probe (default: instruct base).",
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=DEFAULT_DURATION_MINUTES,
        help="Total runtime window for the injector (recommended range: 30-60 minutes).",
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=DEFAULT_INTERACTIONS,
        help="Number of scripted turns per profile when using the bundled defaults (default: 5).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Minimum cool-down between probe iterations (default: 30s).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help=f"Directory to store probe JSONL logs (default: {DEFAULT_LOG_DIR.as_posix()}).",
    )
    parser.add_argument(
        "--log-file",
        help="Write to a specific JSONL path instead of deriving one per iteration.",
    )
    parser.add_argument(
        "--turn",
        action="append",
        help="Override the default scripted turns; repeat to supply multiple prompts.",
    )
    parser.add_argument(
        "--no-reset-session",
        action="store_false",
        dest="reset_session",
        help="Skip resetting live session state between iterations.",
    )
    parser.set_defaults(reset_session=True)
    parser.add_argument(
        "--no-retain-rows",
        action="store_true",
        help="Only log summaries; skip the per-turn row payloads in the JSONL output.",
    )
    parser.add_argument(
        "--lock-file",
        default=str(cp.LOCK_PATH_DEFAULT),
        help="Guard file to prevent concurrent probe injectors (default mirrors continuous probes).",
    )
    return parser.parse_args()


def _next_harness_directory(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = base / f"harness-{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


async def _run_mid_span(args: argparse.Namespace) -> None:
    turns = _resolve_turns(args)
    duration_minutes = args.duration_minutes
    if duration_minutes <= 0:
        raise SystemExit("--duration-minutes must be positive.")
    cooldown = max(0.0, args.cooldown_seconds)
    prep_start = datetime.now().astimezone()
    lock = cp._ProbeLock(Path(args.lock_file))
    deadline: datetime | None = None
    loop_start: datetime | None = None
    iteration = 0
    completed = 0
    failed = 0
    stop_reason = "duration reached"
    run_dir: Path | None = None
    log_dir_target = args.log_dir
    if not args.log_file:
        run_dir = _next_harness_directory(Path(args.log_dir))
        log_dir_target = str(run_dir)
        print(f"[mid_span_probes] Using run directory: {run_dir}")
    totals = defaultdict(int)
    high_self_streaks = defaultdict(int)
    current_self_streaks = defaultdict(int)
    low_self_hits = defaultdict(int)
    auth_below_hits = defaultdict(int)
    gate_passes = defaultdict(int)
    auto_tweaks_applied: set[str] = set()
    summary_rows: list[dict[str, object]] = []

    try:
        lock.acquire()
        if args.reset_session:
            await cp._reset_live_session("mid_span_probe_boot")
        loop_start = datetime.now().astimezone()
        deadline = loop_start + timedelta(minutes=duration_minutes)
        print(
            "[mid_span_probes] Requested runtime: %.1f minutes "
            "(window start %s, deadline %s)"
            % (
                duration_minutes,
                loop_start.isoformat(timespec="seconds"),
                deadline.isoformat(timespec="seconds"),
            )
        )
        while True:
            now = datetime.now().astimezone()
            if deadline and now >= deadline:
                break
            iteration += 1
            batch_time = datetime.now(timezone.utc)
            log_path = cp._resolve_log_path(args.log_file, log_dir_target, batch_time, iteration)
            print(f"[mid_span_probes] Iteration {iteration}; logging to {log_path}")
            if args.reset_session and iteration > 1:
                await cp._reset_live_session(f"mid_span_probe_iter_{iteration}")
            try:
                probe_results = await cp._run_profiles(
                    profiles=args.profiles,
                    turns=turns,
                    log_path=log_path,
                    retain_rows=not args.no_retain_rows,
                )
            except Exception as exc:
                failed += 1
                trace_payload = {
                    "timestamp": cp._now_iso(),
                    "error": str(exc),
                    "iteration": iteration,
                    "traceback": traceback.format_exc(),
                }
                cp._log_probe_result(log_path, trace_payload)
                print(f"[mid_span_probes] Iteration {iteration} failed: {exc}. Logged error and continuing.")
                await cp._reset_live_session(f"mid_span_probe_iter_{iteration}_error")
            else:
                completed += 1
                for profile, probe in zip(args.profiles, probe_results):
                    summary = probe.get("summary") or {}
                    self_val = float(summary.get("self_preoccupation", 0.0) or 0.0)
                    auth_val = float(summary.get("authenticity_score", 0.0) or 0.0)
                    drift_val = float(summary.get("assistant_drift", 0.0) or 0.0)
                    gate = summary.get("gate") or {}
                    totals[profile] += 1
                    summary_rows.append(
                        {
                            "iteration": iteration,
                            "profile": profile,
                            "timestamp": summary.get("timestamp", cp._now_iso()),
                            "authenticity": auth_val,
                            "self_preoccupation": self_val,
                            "assistant_drift": drift_val,
                            "outward_streak": float(summary.get("outward_streak_score", 0.0) or 0.0),
                            "gate": gate,
                        }
                    )
                    if self_val > 0.75:
                        current_self_streaks[profile] += 1
                        high_self_streaks[profile] = max(
                            high_self_streaks[profile], current_self_streaks[profile]
                        )
                        if current_self_streaks[profile] >= 5:
                            print(
                                f"[mid_span_probes][warn] profile={profile} high self streak "
                                f"{current_self_streaks[profile]} (self={self_val:.3f})"
                            )
                    else:
                        if current_self_streaks[profile] >= 5:
                            print(
                                f"[mid_span_probes][info] profile={profile} high-self streak ended "
                                f"at self={self_val:.3f}"
                            )
                        current_self_streaks[profile] = 0
                    if self_val < 0.4:
                        low_self_hits[profile] += 1
                    if auth_val < 0.45:
                        auth_below_hits[profile] += 1
                    auth_gate_ok = bool(((gate.get("authenticity_score") or {}).get("ok")))
                    self_gate_ok = bool(((gate.get("self_preoccupation") or {}).get("ok")))
                    if auth_gate_ok and self_gate_ok:
                        gate_passes[profile] += 1
                    if (
                        iteration >= 10
                        and gate_passes[profile] == 0
                        and profile not in auto_tweaks_applied
                    ):
                        _apply_auto_tweak(profile, iteration, log_path)
                        auto_tweaks_applied.add(profile)

            now = datetime.now().astimezone()
            if deadline and now >= deadline:
                break
            remaining = (deadline - now).total_seconds() if deadline else 0.0
            sleep_seconds = min(cooldown, remaining)
            if sleep_seconds <= 0:
                continue
            await _cooldown_sleep(sleep_seconds)
    finally:
        try:
            await app_main._shutdown_clients()
        except AttributeError:
            pass
        lock.release()
        loop_end = datetime.now().astimezone()
        start_clock = loop_start or prep_start
        elapsed = loop_end - start_clock
        if totals:
            print("[mid_span_probes] Profile summary:")
            for profile in args.profiles:
                total = totals.get(profile, 0)
                if not total:
                    continue
                streak = high_self_streaks.get(profile, 0)
                lows = low_self_hits.get(profile, 0)
                auth_low = auth_below_hits.get(profile, 0)
                print(
                    f"  {profile}: samples={total} high_self_streak={streak} "
                    f"self<0.4={lows} auth<0.45={auth_low}"
                )
        if summary_rows and run_dir:
            summary_path = run_dir / "summary_compact.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary_rows, handle, indent=2)
            print(f"[mid_span_probes] Wrote summary to {summary_path}")
            meta = {
                "requested_minutes": duration_minutes,
                "loop_start": loop_start.isoformat(timespec="seconds") if loop_start else None,
                "loop_end": loop_end.isoformat(timespec="seconds"),
                "deadline": deadline.isoformat(timespec="seconds") if deadline else None,
                "elapsed_seconds": elapsed.total_seconds(),
                "target_seconds": float(duration_minutes) * 60.0,
                "iterations_completed": completed,
                "iterations_failed": failed,
            }
            meta_path = run_dir / "run_meta.json"
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
            print(f"[mid_span_probes] Wrote run metadata to {meta_path}")
        print(
            "[mid_span_probes] Summary: elapsed=%s, completed=%d, failed=%d, stop_reason=%s"
            % (cp._format_elapsed(elapsed), completed, failed, stop_reason)
        )


def main() -> None:
    args = parse_args()
    asyncio.run(_run_mid_span(args))


if __name__ == "__main__":
    main()
