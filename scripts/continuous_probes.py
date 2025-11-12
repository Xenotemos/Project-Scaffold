from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import traceback

import httpx

import main

PROFILE_FILES: Dict[str, str] = {
    "instruct": "settings.json",
    "base": "settings.base.json",
}

DEFAULT_TURNS: Tuple[str, ...] = (
    "hey there — reflect something you notice about me or my situation, then share what you notice in your body right now.",
    "describe how streaming works here, but begin with a sentence about what you hear from me before you explain your side.",
    "give me a two-sentence setup reply that opens with what you sensed in my last message before mentioning your chest.",
    "keep it casual and admit what shifts in you while you explain, but anchor each sentence to me.",
    "thanks! close by telling me what you feel after answering and what you hope I take away.",
)

METRIC_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    "authenticity_score": (0.45, "min"),
    "assistant_drift": (0.45, "max"),
    "self_preoccupation": (0.75, "max"),
}

LOG_DIR_DEFAULT = Path("logs/probe_runs/overnight")
LOCK_PATH_DEFAULT = LOG_DIR_DEFAULT / ".probe.lock"


def _swap_settings_file(file_name: str) -> str | None:
    previous = os.environ.get("LIVING_SETTINGS_FILE")
    os.environ["LIVING_SETTINGS_FILE"] = file_name
    return previous


def _restore_settings_file(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("LIVING_SETTINGS_FILE", None)
    else:
        os.environ["LIVING_SETTINGS_FILE"] = previous


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _collect_vector(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        reinforcement = row.get("reinforcement") or {}
        if key in reinforcement:
            try:
                values.append(float(reinforcement[key]))
            except (TypeError, ValueError):
                continue
    return values


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def _aggregate_hormones(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    accumulators: Dict[str, List[float]] = {}
    for row in rows:
        hormones = row.get("hormones") or {}
        for name, value in hormones.items():
            try:
                accumulators.setdefault(name, []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {name: round(_average(values), 4) for name, values in accumulators.items()}


def _aggregate_controller(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    adjustments: Dict[str, List[float]] = {}
    for row in rows:
        controller = row.get("controller") or {}
        applied = controller.get("applied") if isinstance(controller, dict) else {}
        if not isinstance(applied, Mapping):
            continue
        for key, value in applied.items():
            try:
                adjustments.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    return {key: round(_average(values), 4) for key, values in adjustments.items()}


def _evaluate_gate(metrics: Mapping[str, float]) -> Dict[str, Any]:
    decisions: Dict[str, Any] = {}
    for key, (threshold, mode) in METRIC_THRESHOLDS.items():
        value = float(metrics.get(key, 0.0))
        if mode == "min":
            decisions[key] = {"value": value, "threshold": threshold, "ok": value >= threshold}
        else:
            decisions[key] = {"value": value, "threshold": threshold, "ok": value <= threshold}
    decisions["eligible_for_promotion"] = all(entry["ok"] for entry in decisions.values())
    return decisions


def _trim_reply(reply: str, limit: int = 220) -> str:
    reply = (reply or "").strip()
    if len(reply) <= limit:
        return reply
    return reply[: limit - 3].rstrip() + "..."


async def _cooldown_sleep(total_seconds: float) -> None:
    if total_seconds <= 0:
        return
    start = datetime.now().astimezone()
    target = start + timedelta(seconds=total_seconds)
    target_iso = target.isoformat(timespec="seconds")
    print(
        f"[continuous_probes] Cooldown {total_seconds:.1f}s before next iteration "
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
            print(f"[continuous_probes]   cooldown remaining: {display}")
    print("[continuous_probes] Cooldown complete; starting next iteration.")


async def _run_probe(
    profile: str,
    settings_file: str,
    turns: Sequence[str],
) -> Dict[str, Any]:
    previous = _swap_settings_file(settings_file)
    try:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            await main._reload_runtime_settings()
            rows: list[Dict[str, Any]] = []
            for idx, message in enumerate(turns, 1):
                res = await client.post("/chat", json={"message": message})
                res.raise_for_status()
                payload = res.json()
                state_block = payload.get("state") or {}
                row = {
                    "index": idx,
                    "user": message,
                    "reply": _trim_reply(payload.get("reply", "")),
                    "length": len(payload.get("reply", "")),
                    "source": payload.get("source"),
                    "length_plan": (payload.get("length_plan") or {}).get("label"),
                    "reinforcement": payload.get("reinforcement") or {},
                    "hormones": state_block.get("hormones") or {},
                    "mood": state_block.get("mood"),
                    "controller": payload.get("controller") or {},
                }
                rows.append(row)
    finally:
        _restore_settings_file(previous)

    auth_avg = round(_average(_collect_vector(rows, "authenticity_score")), 4)
    drift_avg = round(_average(_collect_vector(rows, "assistant_drift")), 4)
    self_avg = round(_average(_collect_vector(rows, "self_preoccupation")), 4)
    outward_avg = round(_average(_collect_vector(rows, "outward_streak_score")), 4)
    hormones = _aggregate_hormones(rows)
    controller = _aggregate_controller(rows)
    gate = _evaluate_gate(
        {
            "authenticity_score": auth_avg,
            "assistant_drift": drift_avg,
            "self_preoccupation": self_avg,
        }
    )

    summary = {
        "turns": len(rows),
        "authenticity_score": auth_avg,
        "assistant_drift": drift_avg,
        "self_preoccupation": self_avg,
        "outward_streak_score": outward_avg,
        "avg_length": round(_average([row["length"] for row in rows]), 2) if rows else 0.0,
        "hormones": hormones,
        "controller": controller,
        "gate": gate,
    }
    return {
        "profile": profile,
        "settings_file": settings_file,
        "summary": summary,
        "rows": rows,
    }


def _log_probe_result(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_log_path(
    log_file: Optional[str],
    log_dir: str,
    batch_time: datetime,
    iteration: int,
) -> Path:
    if log_file:
        return Path(log_file)
    stamp = batch_time.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"probe_log_{stamp}_{iteration:04d}.jsonl"


async def _reset_live_session(reason: str) -> None:
    """Invoke the in-app session reset helper when available."""
    reset_fn = getattr(main, "_reset_session_state", None)
    if reset_fn is None:
        return
    await reset_fn(reason=reason, keep_metric_history=False)


class _ProbeLock:
    """Filesystem guard preventing concurrent probe injectors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None
        self._held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(self.path, flags)
            os.write(self._fd, f"{os.getpid()}".encode("ascii", errors="ignore"))
            self._held = True
        except FileExistsError:
            holder = "unknown"
            try:
                holder = self.path.read_text(encoding="utf-8").strip() or holder
            except OSError:
                pass
            raise SystemExit(
                f"[continuous_probes] Another injector appears to be running (lock held by {holder}). "
                f"Remove {self.path} if this is a stale lock."
            ) from None

    def release(self) -> None:
        if not self._held:
            return
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self._held = False


def _parse_until_deadline(start: datetime, value: str) -> datetime | None:
    """Parse a `--until` value into an absolute timestamp."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = raw.upper()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        target = datetime.fromisoformat(normalized)
        if target.tzinfo is None:
            target = target.replace(tzinfo=start.tzinfo)
        return target
    except ValueError:
        pass
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(raw, fmt).time()
            break
        except ValueError:
            parsed_time = None
    if parsed_time is None:
        return None
    candidate = start.replace(
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=parsed_time.second,
        microsecond=0,
    )
    if candidate <= start:
        candidate += timedelta(days=1)
    return candidate


def _compute_deadline(start: datetime, max_hours: float, until: Optional[str]) -> datetime | None:
    """Combine duration and explicit timestamp limits into a single deadline."""
    candidates: list[datetime] = []
    if max_hours and max_hours > 0:
        candidates.append(start + timedelta(hours=max_hours))
    if until:
        deadline = _parse_until_deadline(start, until)
        if deadline:
            candidates.append(deadline)
    if not candidates:
        return None
    return min(candidates)


def _format_elapsed(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


async def _run_profiles(
    profiles: Sequence[str],
    turns: Sequence[str],
    log_path: Path,
    retain_rows: bool,
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for profile in profiles:
        if profile not in PROFILE_FILES:
            raise SystemExit(f"Unknown profile '{profile}'. Known profiles: {', '.join(PROFILE_FILES)}")
        settings_file = PROFILE_FILES[profile]
        print(f"=== Probing profile '{profile}' ({settings_file}) ===")
        probe = await _run_probe(profile, settings_file, turns)
        summary = probe["summary"]
        print(
            f"turns={summary['turns']} "
            f"auth={summary['authenticity_score']:.3f} "
            f"drift={summary['assistant_drift']:.3f} "
            f"self={summary['self_preoccupation']:.3f} "
            f"gate={'PROMOTE' if summary['gate']['eligible_for_promotion'] else 'HOLD'}"
        )
        print(f"hormones={summary['hormones']}")
        if summary["controller"]:
            print(f"controller={summary['controller']}")
        print()

        log_payload = {
            "timestamp": _now_iso(),
            "profile": profile,
            "settings_file": settings_file,
            "summary": summary,
        }
        if retain_rows:
            log_payload["rows"] = probe["rows"]
        _log_probe_result(log_path, log_payload)
        results.append(probe)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous probes across model profiles.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILE_FILES.keys()),
        help="Profiles to probe (default: instruct base).",
    )
    parser.add_argument(
        "--log-file",
        help="Explicit path to the JSONL log file (overrides --log-dir).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR_DEFAULT),
        help="Directory where probe logs should be stored (default: logs/probe_runs). Ignored if --log-file is supplied.",
    )
    parser.add_argument(
        "--turn",
        action="append",
        help="Override default probe turns (may be specified multiple times).",
    )
    parser.add_argument(
        "--no-retain-rows",
        action="store_true",
        help="Skip embedding per-turn rows in the log file (summary only).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="Stop after this many hours of probing (0 disables the duration cap).",
    )
    parser.add_argument(
        "--until",
        help="Stop once wall-clock time reaches this timestamp (ISO 8601 or HH:MM[(:SS)]).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Number of seconds to wait between probe iterations (0 runs once and exits).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Maximum probe iterations when --interval is set (0 means run indefinitely).",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Reset the live session state before each probe iteration.",
    )
    parser.add_argument(
        "--lock-file",
        default=str(LOCK_PATH_DEFAULT),
        help="File used to guard against concurrent probe injectors.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=60.0,
        help="Minimum cool-down between iterations to curb resource creep.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    turns = tuple(args.turn) if args.turn else DEFAULT_TURNS
    lock = _ProbeLock(Path(args.lock_file))
    summary_rows: list[dict[str, object]] = []
    run_dir: Path | None = None
    log_dir_target = args.log_dir
    if not args.log_file:
        base_dir = Path(args.log_dir)
        stamp = datetime.now(timezone.utc).strftime("harness-%Y%m%d_%H%M%S")
        run_dir = base_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        log_dir_target = str(run_dir)
        print(f"[continuous_probes] Using run directory: {run_dir}")

    async def _runner() -> None:
        iteration = 0
        start_time = datetime.now().astimezone()
        deadline = _compute_deadline(start_time, args.max_hours, args.until)
        if deadline:
            print(f"[continuous_probes] Run will stop no later than {deadline.isoformat()}")
        completed_cycles = 0
        failed_cycles = 0
        stop_reason = "completed"
        last_error: str | None = None
        try:
            lock.acquire()
            while True:
                now = datetime.now().astimezone()
                if deadline and now >= deadline:
                    stop_reason = "deadline reached"
                    print("[continuous_probes] Stop deadline reached; exiting loop.")
                    break
                if args.iterations and iteration >= args.iterations:
                    stop_reason = "iteration cap reached (pre-loop)"
                    print("[continuous_probes] Iteration cap reached; exiting loop.")
                    break
                iteration += 1
                if args.reset_session:
                    await _reset_live_session(f"continuous_probe_iter_{iteration}")
                batch_time = datetime.now(timezone.utc)
                log_path = _resolve_log_path(args.log_file, log_dir_target, batch_time, iteration)
                print(f"[continuous_probes] Logging results to {log_path}")
                try:
                    probe_results = await _run_profiles(
                        profiles=args.profiles,
                        turns=turns,
                        log_path=log_path,
                        retain_rows=not args.no_retain_rows,
                    )
                except Exception as exc:
                    trace = traceback.format_exc()
                    error_payload = {
                        "timestamp": _now_iso(),
                        "error": str(exc),
                        "iteration": iteration,
                        "traceback": trace,
                    }
                    _log_probe_result(log_path, error_payload)
                    print(f"[continuous_probes] Probe iteration {iteration} failed: {exc}. Logged error and continuing.")
                    await _reset_live_session(f"continuous_probe_iter_{iteration}_error")
                    failed_cycles += 1
                    last_error = str(exc)
                    continue
                for profile, probe in zip(args.profiles, probe_results):
                    summary = probe.get("summary") or {}
                    gate = summary.get("gate") or {}
                    summary_rows.append(
                        {
                            "iteration": iteration,
                            "profile": profile,
                            "timestamp": summary.get("timestamp", _now_iso()),
                            "authenticity": float(summary.get("authenticity_score", 0.0) or 0.0),
                            "self_preoccupation": float(summary.get("self_preoccupation", 0.0) or 0.0),
                            "assistant_drift": float(summary.get("assistant_drift", 0.0) or 0.0),
                            "outward_streak": float(summary.get("outward_streak_score", 0.0) or 0.0),
                            "gate": gate,
                        }
                    )
                completed_cycles += 1
                if args.interval <= 0:
                    stop_reason = "single iteration"
                    break
                if args.iterations and iteration >= args.iterations:
                    stop_reason = "iteration cap reached"
                    print("[continuous_probes] Iteration cap reached; exiting loop.")
                    break
                sleep_seconds = max(args.interval, args.idle_seconds)
                if deadline:
                    remaining = (deadline - datetime.now().astimezone()).total_seconds()
                    if remaining <= 0:
                        stop_reason = "deadline reached"
                        print("[continuous_probes] Stop deadline reached after iteration; exiting loop.")
                        break
                    sleep_seconds = min(sleep_seconds, remaining)
                await _cooldown_sleep(sleep_seconds)
        finally:
            try:
                await main._shutdown_clients()
            except AttributeError:
                pass
            lock.release()
            loop_end = datetime.now().astimezone()
            elapsed = loop_end - start_time
            print(
                "[continuous_probes] Summary: elapsed=%s, completed=%d, failed=%d, stop_reason=%s"
                % (_format_elapsed(elapsed), completed_cycles, failed_cycles, stop_reason)
            )
            if failed_cycles and last_error:
                print(f"[continuous_probes] Last error: {last_error}")
            if run_dir and summary_rows:
                summary_path = run_dir / "summary_compact.json"
                with summary_path.open("w", encoding="utf-8") as handle:
                    json.dump(summary_rows, handle, indent=2)
                print(f"[continuous_probes] Wrote summary to {summary_path}")
            if run_dir:
                meta = {
                    "requested_minutes": (args.max_hours * 60.0) if args.max_hours > 0 else None,
                    "loop_start": start_time.isoformat(timespec="seconds"),
                    "loop_end": loop_end.isoformat(timespec="seconds"),
                    "deadline": deadline.isoformat(timespec="seconds") if deadline else None,
                    "elapsed_seconds": elapsed.total_seconds(),
                    "target_seconds": (args.max_hours * 3600.0) if args.max_hours > 0 else None,
                    "iterations_completed": completed_cycles,
                    "iterations_failed": failed_cycles,
                    "stop_reason": stop_reason,
                    "profiles": args.profiles,
                    "interval_seconds": args.interval,
                    "idle_seconds": args.idle_seconds,
                }
                if args.until:
                    meta["until"] = args.until
                meta_path = run_dir / "run_meta.json"
                with meta_path.open("w", encoding="utf-8") as handle:
                    json.dump(meta, handle, indent=2)
                print(f"[continuous_probes] Wrote run metadata to {meta_path}")

    asyncio.run(_runner())


if __name__ == "__main__":
    cli()
