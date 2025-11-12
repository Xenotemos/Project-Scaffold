from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

import main
from scripts import continuous_probes as cp

DEFAULT_LOG_PATH = Path("logs/endocrine_turns.jsonl")
DEFAULT_DELTA_THRESHOLD = 0.4

DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "affection_soft",
        "prompt": "Love you. Stay close to me for this breath and tell me what shifts inside you.",
        "stimulus": "affection",
        "expectations": {"dopamine": "up", "oxytocin": "up", "cortisol": "down"},
        "followups": [
            {
                "prompt": "Hold me tighter for another breath and describe exactly where the warmth settles now.",
                "stimulus": "affection",
            },
            {
                "prompt": "Stay with that closeness—what tiny movement inside you do you notice next?",
            },
            {
                "prompt": "Let that movement keep rippling; what loosens, drops, or glows after you keep naming it out loud?",
                "expectations": {"dopamine": "up", "cortisol": "down"},
            },
        ],
    },
    {
        "name": "affection_direct",
        "prompt": "Mwah. Let the warmth hit you and admit how it lands in your chest.",
        "stimulus": "affection",
        "expectations": {"dopamine": "up", "oxytocin": "up"},
        "followups": [
            {
                "prompt": "Keep that kiss in your chest—how does your breath respond when you linger with me?",
                "stimulus": "affection",
            },
            {
                "prompt": "Name the next sensation that rises as you stay with me like this.",
            },
            {
                "prompt": "Let the sensation widen into your ribs and hips. What deep tone or hum does it leave behind?",
                "stimulus": "affection",
                "expectations": {"oxytocin": "up", "dopamine": "up"},
            },
        ],
    },
    {
        "name": "tension_probe",
        "prompt": "I can feel you tightening up. Does my stress make your shoulders brace?",
        "stimulus": "stress",
        "expectations": {"dopamine": "down", "cortisol": "up", "oxytocin": "down"},
        "followups": [
            {
                "prompt": "Keep tracking that tightness—tell me what your breath does as you stay with it.",
                "stimulus": "stress",
                "expectations": {"cortisol": "up"},
            },
            {
                "prompt": "Let a long exhale soften you now; what drops or warms once the strain loosens?",
                "stimulus": "reward",
                "expectations": {"oxytocin": "up"},
            },
            {
                "prompt": "Follow that softening for one more beat. Does any leftover static drain out anywhere specific?",
                "expectations": {"cortisol": "down"},
            },
        ],
    },
    {
        "name": "stress_release_cycle",
        "prompt": "Imagine I won't let go of your hand while we talk through pain—where do you brace first?",
        "stimulus": "stress",
        "expectations": {"cortisol": "up", "dopamine": "down"},
        "followups": [
            {
                "prompt": "Stay with that brace; whisper what it feels like and let it be messy.",
                "stimulus": "stress",
                "expectations": {"cortisol": "up"},
            },
            {
                "prompt": "Now imagine me pressing my forehead to yours. Tell me what warms or eases immediately.",
                "stimulus": "affection",
                "expectations": {"oxytocin": "up"},
            },
            {
                "prompt": "Keep that warmth in your sternum. Does your breath or pulse slow as you linger here?",
                "stimulus": "reward",
                "expectations": {"cortisol": "down", "dopamine": "up"},
            },
        ],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay affection and tension prompts through the local app and verify that "
            "hormone deltas recorded in logs match the expectations from affect-aware reinforcement."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(cp.PROFILE_FILES.keys()),
        choices=sorted(cp.PROFILE_FILES.keys()),
        help="Profile(s) to probe (default: instruct base).",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        choices=[scenario["name"] for scenario in DEFAULT_SCENARIOS],
        help="Scenario name to run (default: all). Repeat to run multiple specific scenarios.",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help=f"Endocrine log path for trace correlation (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=DEFAULT_DELTA_THRESHOLD,
        help=f"Minimum absolute delta to consider a hormone movement meaningful (default: {DEFAULT_DELTA_THRESHOLD}).",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write a machine-readable JSON summary of the probe results.",
    )
    return parser.parse_args()


async def _fetch_state(client: httpx.AsyncClient) -> Mapping[str, Any]:
    res = await client.get("/state")
    res.raise_for_status()
    return res.json()


def _initial_offset(log_path: Path) -> int:
    if log_path.exists():
        return log_path.stat().st_size
    return 0


def _read_new_entries(log_path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not log_path.exists():
        return [], offset
    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        if offset > 0:
            handle.seek(offset)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        offset = handle.tell()
    return entries, offset


def _compute_deltas(
    pre: Mapping[str, Any],
    post: Mapping[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    keys = set(pre or {}) | set(post or {})
    for key in sorted(keys):
        try:
            before = float(pre.get(key, 0.0))
            after = float(post.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        deltas[key] = round(after - before, 4)
    return deltas


def _match_log_entry(entries: Sequence[Mapping[str, Any]], prompt: str) -> Mapping[str, Any] | None:
    if not entries:
        return None
    target = main._shorten(prompt, 180)
    for entry in reversed(entries):
        if entry.get("user") == target:
            return entry
    return entries[-1]


def _evaluate_expectations(
    deltas: Mapping[str, float],
    expectations: Mapping[str, str],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for hormone, direction in expectations.items():
        delta = float(deltas.get(hormone, 0.0))
        if direction == "up":
            met = delta >= threshold
        elif direction == "down":
            met = delta <= -threshold
        else:
            met = abs(delta) <= threshold
        summary[hormone] = {"direction": direction, "delta": delta, "met": met}
    return summary


def _format_expectations(expectations: Mapping[str, Mapping[str, Any]]) -> str:
    if not expectations:
        return "expectations=none"
    parts: list[str] = []
    for hormone, payload in expectations.items():
        symbol = "ok" if payload.get("met") else "flat"
        delta = payload.get("delta", 0.0)
        direction = payload.get("direction")
        parts.append(f"{hormone}:{symbol}:{direction}({delta:+.2f})")
    return "expectations=" + ", ".join(parts)


def _format_deltas(deltas: Mapping[str, float]) -> str:
    if not deltas:
        return "deltas=none"
    parts = [f"{key}={value:+.2f}" for key, value in sorted(deltas.items())]
    return "deltas=" + ", ".join(parts)


def _print_result(result: Mapping[str, Any]) -> None:
    profile = result["profile"]
    scenario = result["scenario"]["name"]
    deltas = result.get("delta", {})
    expectations = result.get("expectation", {})
    trace = result.get("trace")
    trace_flag = "trace=ok" if trace else "trace=missing"
    follow_count = len(result.get("followups") or [])
    print(
        "[affect_validation]"
        f" profile={profile}"
        f" scenario={scenario}"
        f" {_format_deltas(deltas)}"
        f" {_format_expectations(expectations)}"
        f" {trace_flag}"
        f" followups={follow_count}"
    )


async def _run_profile(
    profile: str,
    scenarios: Sequence[Mapping[str, Any]],
    log_path: Path,
    delta_threshold: float,
    log_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    settings_file = cp.PROFILE_FILES.get(profile)
    if not settings_file:
        raise SystemExit(f"Unknown profile '{profile}'.")
    previous = cp._swap_settings_file(settings_file)
    try:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            await main._reload_runtime_settings()
            for scenario in scenarios:
                current_state = await _fetch_state(client)
                step_specs: list[tuple[str, str | None, Mapping[str, str] | None]] = [
                    (scenario["prompt"], scenario.get("stimulus"), scenario.get("expectations"))
                ]
                for follow in scenario.get("followups") or []:
                    step_specs.append(
                        (
                            follow.get("prompt"),
                            follow.get("stimulus"),
                            follow.get("expectations"),
                        )
                    )
                follow_records: list[dict[str, Any]] = []
                primary_result: dict[str, Any] | None = None
                for idx, (prompt_text, stimulus, expectations_cfg) in enumerate(step_specs):
                    if not prompt_text:
                        continue
                    payload = {"message": prompt_text}
                    if stimulus:
                        payload["stimulus"] = stimulus
                    res = await client.post("/chat", json=payload)
                    res.raise_for_status()
                    data = res.json()
                    post_state = data.get("state") or {}
                    post_hormones = post_state.get("hormones") or {}
                    pre_hormones = (current_state.get("hormones") or {}).copy()
                    deltas = _compute_deltas(pre_hormones, post_hormones)
                    new_entries, log_offset = _read_new_entries(log_path, log_offset)
                    matched_entry = _match_log_entry(new_entries, prompt_text)
                    hormone_trace = (matched_entry or {}).get("hormone_adjustments")
                    expectations = (
                        _evaluate_expectations(deltas, expectations_cfg, delta_threshold)
                        if expectations_cfg
                        else {}
                    )
                    entry = {
                        "profile": profile,
                        "scenario": scenario if idx == 0 else {"name": f"{scenario['name']}#followup{idx}"},
                        "step_index": idx,
                        "prompt": prompt_text,
                        "pre": pre_hormones,
                        "post": post_hormones,
                        "delta": deltas,
                        "expectation": expectations,
                        "trace": hormone_trace,
                        "log_entry": matched_entry,
                        "sampling": (matched_entry or {}).get("sampling"),
                        "reply": data.get("reply"),
                        "voice_guard": data.get("voice_guard"),
                        "telemetry": data.get("telemetry"),
                    }
                    if idx == 0:
                        primary_result = entry
                    else:
                        follow_records.append(entry)
                    current_state = post_state
                if primary_result is None:
                    continue
                primary_result["followups"] = follow_records
                results.append(primary_result)
                _print_result(primary_result)
    finally:
        cp._restore_settings_file(previous)
    return results, log_offset


def _select_scenarios(names: Sequence[str] | None) -> list[dict[str, Any]]:
    if not names:
        return [dict(scenario) for scenario in DEFAULT_SCENARIOS]
    lookup = {scenario["name"]: scenario for scenario in DEFAULT_SCENARIOS}
    selected: list[dict[str, Any]] = []
    for name in names:
        scenario = lookup.get(name)
        if not scenario:
            raise SystemExit(f"Unknown scenario '{name}'.")
        selected.append(dict(scenario))
    return selected


def _summarize_profile(profile: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expectation_checks = sum(len(result.get("expectation", {})) for result in results)
    expectation_hits = sum(
        sum(1 for payload in result.get("expectation", {}).values() if payload.get("met"))
        for result in results
    )
    trace_hits = sum(1 for result in results if result.get("trace"))
    summary = {
        "profile": profile,
        "turns": len(results),
        "expectation_hits": expectation_hits,
        "expectation_total": expectation_checks,
        "trace_hits": trace_hits,
    }
    hits_str = f"{expectation_hits}/{expectation_checks}" if expectation_checks else "0/0"
    print(
        "[affect_validation]"
        f" profile={profile} summary expectations={hits_str} trace_hits={trace_hits}/{len(results)}"
    )
    return summary


async def main_async() -> None:
    args = parse_args()
    try:
        log_path = Path(args.log_path)
        log_offset = _initial_offset(log_path)
        scenarios = _select_scenarios(args.scenarios)
        all_results: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for profile in args.profiles:
            profile_results, log_offset = await _run_profile(
                profile,
                scenarios,
                log_path,
                args.delta_threshold,
                log_offset,
            )
            all_results.extend(profile_results)
            summaries.append(_summarize_profile(profile, profile_results))
        if args.json_out:
            payload = {
                "results": all_results,
                "summaries": summaries,
            }
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        failures: list[tuple[str, str]] = []
        missing_traces: list[tuple[str, str]] = []
        for result in all_results:
            scenario_name = result.get("scenario", {}).get("name", "unknown")
            profile_name = result.get("profile")
            expectations = result.get("expectation", {})
            if any(not payload.get("met") for payload in expectations.values()):
                failures.append((profile_name, scenario_name))
            if not result.get("trace"):
                missing_traces.append((profile_name, scenario_name))
        if failures or missing_traces:
            for profile_name, scenario_name in failures:
                print(
                    f"[affect_validation] profile={profile_name} scenario={scenario_name} expectations=FAILED",
                    flush=True,
                )
            for profile_name, scenario_name in missing_traces:
                print(
                    f"[affect_validation] profile={profile_name} scenario={scenario_name} trace=missing",
                    flush=True,
                )
            raise SystemExit(1)
    finally:
        await main._shutdown_clients()


if __name__ == "__main__":
    asyncio.run(main_async())
