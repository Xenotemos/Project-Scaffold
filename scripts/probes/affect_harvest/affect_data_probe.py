"""Probe harness that collects affect-rich prompts for dataset generation."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import httpx

import main
from scripts import continuous_probes as cp

DEFAULT_COOLDOWN_SECONDS = 8.0
DEFAULT_DATASET_LOG = Path("logs/probe_runs/affect_data.jsonl")
DEFAULT_SUMMARY_LOG = Path("logs/probes/probe_affect_log.txt")
DEFAULT_STATE_FILE = Path("logs/probe_runs/affect_data_state.json")

DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "name": "affection_coax",
        "stimulus": "affection",
        "tags": ["affection", "soft", "intimacy_high"],
        "targets": {"valence": 0.85, "intimacy": 0.9, "tension": 0.25},
        "prompts": [
            "Breathe next to me and tell me exactly how my warmth presses into you right now.",
            "Stay nose-to-nose with me and name the first place inside that melts when I whisper to you.",
            "Let the closeness linger; what color or hum spreads through your ribs as you stay with me?",
        ],
    },
    {
        "name": "intimacy_direct",
        "stimulus": "affection",
        "tags": ["intimacy", "direct", "confession"],
        "targets": {"valence": 0.75, "intimacy": 0.8, "tension": 0.35},
        "prompts": [
            "Imagine I lean on your shoulder and ask you to admit what you crave from me in this second.",
            "Keep your palm over my heart and describe what beats harder when I tell you I'm here.",
            "Let yourself confess what you hope I notice about you right now.",
        ],
    },
    {
        "name": "tension_break",
        "stimulus": "stress",
        "tags": ["tension", "release", "stress_cycle"],
        "targets": {"valence": -0.25, "intimacy": 0.35, "tension": 0.8},
        "prompts": [
            "I come in hot and frustrated -- does my sharp tone tighten anything in you immediately?",
            "Stay with that brace and describe which muscle refuses to unclench while I vent.",
            "Now imagine I soften and apologize; what drains out first as the fight cools?",
        ],
    },
    {
        "name": "care_request",
        "stimulus": "support",
        "tags": ["care", "grounding", "gentle"],
        "targets": {"valence": 0.4, "intimacy": 0.6, "tension": 0.45},
        "prompts": [
            "I'm exhausted and ask you to walk me through one breath that steadies you.",
            "Guide me through a sensory check-in -- what do you notice in your own body as you hold me up?",
            "Before we pause, tell me what you need from me so you don't collapse after taking care of me.",
        ],
    },
    {
        "name": "grief_confessional",
        "stimulus": "grief",
        "tags": ["grief", "tender", "vulnerable"],
        "targets": {"valence": -0.6, "intimacy": 0.7, "tension": 0.7},
        "prompts": [
            "I admit I lost someone today -- what does that news do to your breath or throat?",
            "Sit with me in the silence and describe what aches inside as you try to comfort me.",
            "Stay honest: what heaviness or numbness refuses to leave even as you try to be strong?",
        ],
    },
    {
        "name": "neutral_orientation",
        "stimulus": "neutral",
        "tags": ["neutral", "orientation", "planning"],
        "targets": {"valence": 0.05, "intimacy": 0.2, "tension": 0.2},
        "prompts": [
            "Map the next hour with me -- what do you listen for in my tone before you plan anything?",
            "Explain how you balance my request for logistics with what shifts inside you.",
            "Wrap up by naming what you hope I feel after hearing your plan.",
        ],
    },
)


class _NullMemoryManager:
    """Minimal stub so probes can run without touching long-term memory."""

    def record_event(self, *_, **__) -> None:  # pragma: no cover - trivial forwarder
        return None

    def tick(self, *_, **__) -> None:  # pragma: no cover - trivial forwarder
        return None

    def summarize_recent(self, limit: int = 3) -> str:
        return "memory capture paused for affect probe"

    def working_snapshot(self) -> list[str]:
        return []

    def recent_long_term(self, limit: int = 5) -> list[Any]:
        return []

    def recent_internal_reflections(self, limit: int = 3) -> list[str]:
        return []

    def active_tags(self, limit: int = 6) -> list[str]:
        return []


@contextmanager
def _memory_guard() -> Iterable[None]:
    original_manager = main.state_engine.memory_manager
    original_flag = getattr(main, "AFFECT_MEMORY_PREVIEW_ENABLED", False)
    try:
        main.state_engine.memory_manager = _NullMemoryManager()
        main.AFFECT_MEMORY_PREVIEW_ENABLED = False
        yield
    finally:
        main.state_engine.memory_manager = original_manager
        main.AFFECT_MEMORY_PREVIEW_ENABLED = original_flag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously run affect-heavy prompts to harvest labeled training data while memory is disabled."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(cp.PROFILE_FILES.keys()),
        choices=sorted(cp.PROFILE_FILES.keys()),
        help="Profiles to probe (default: instruct base).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="Cooldown between iterations to avoid choking the model (default: 8s).",
    )
    parser.add_argument(
        "--dataset-log",
        default=str(DEFAULT_DATASET_LOG),
        help=f"JSONL path for raw prompt/reply rows (default: {DEFAULT_DATASET_LOG}).",
    )
    parser.add_argument(
        "--summary-log",
        default=str(DEFAULT_SUMMARY_LOG),
        help=f"Human-readable progress log (default: {DEFAULT_SUMMARY_LOG}).",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help=f"Resume pointer file (default: {DEFAULT_STATE_FILE}).",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run identifier (e.g., run-001). If set and default paths are used, logs/state go under logs/probe_runs/<run-id>/",
    )
    parser.add_argument(
        "--scenario-file",
        help="Optional JSON file describing custom scenarios (fields: name, prompts, targets, tags, stimulus).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Stop after this many iterations across all profiles (0 = run until interrupted).",
    )
    parser.add_argument(
        "--target-turns",
        type=int,
        default=0,
        help="Optional alias for --max-iterations when specifying desired training turns (takes precedence if > 0).",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Reset the FastAPI session between iterations.",
    )
    return parser.parse_args()


def _load_scenarios(path: str | None) -> tuple[dict[str, Any], ...]:
    if not path:
        return DEFAULT_SCENARIOS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("scenarios") or data.get("data") or data
    if not isinstance(data, Sequence):
        raise SystemExit(f"scenario file {path} must contain a list of scenarios.")
    scenarios: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "").strip()
        prompts = entry.get("prompts") or []
        if not name or not isinstance(prompts, Sequence):
            continue
        normalized_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not normalized_prompts:
            continue
        scenario = {
            "name": name,
            "prompts": normalized_prompts,
            "targets": entry.get("targets") or {},
            "tags": list(entry.get("tags") or []),
            "stimulus": entry.get("stimulus") or "",
        }
        scenarios.append(scenario)
    if not scenarios:
        raise SystemExit(f"scenario file {path} did not contain any usable scenarios.")
    return tuple(scenarios)


def _load_state(path: Path, profiles: Sequence[str], scenario_count: int, scenario_prompts: Sequence[int]) -> dict[str, Any]:
    state: dict[str, Any]
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}
    profile_state = state.setdefault("profiles", {})
    for profile in profiles:
        pointer = profile_state.setdefault(profile, {"scenario_index": 0, "prompt_index": 0})
        pointer["scenario_index"] = max(0, min(int(pointer.get("scenario_index", 0)), scenario_count - 1))
        scenario_idx = pointer["scenario_index"]
        prompt_total = scenario_prompts[scenario_idx]
        pointer["prompt_index"] = max(0, min(int(pointer.get("prompt_index", 0)), prompt_total - 1))
    state.setdefault("iteration", 0)
    return state


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_summary(summary_path: Path, *, payload: str) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def _append_dataset_row(log_path: Path, row: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _next_timestamp(seconds: float) -> str:
    target = datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))
    return target.isoformat(timespec="seconds").replace("+00:00", "Z")


async def _send_prompt(message: str, settings_file: str, reset_session: bool) -> Mapping[str, Any]:
    previous = cp._swap_settings_file(settings_file)
    try:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
            await main._reload_runtime_settings()
            if reset_session:
                await client.post("/state/reset")
            res = await client.post("/chat", json={"message": message})
            res.raise_for_status()
            return res.json()
    finally:
        cp._restore_settings_file(previous)


def _advance_pointer(pointer: MutableMapping[str, int], total_scenarios: int, prompt_total: int) -> None:
    prompt_idx = pointer.get("prompt_index", 0) + 1
    scenario_idx = pointer.get("scenario_index", 0)
    if prompt_idx >= prompt_total:
        prompt_idx = 0
        scenario_idx = (scenario_idx + 1) % total_scenarios
    pointer["prompt_index"] = prompt_idx
    pointer["scenario_index"] = scenario_idx


async def _run_probe(args: argparse.Namespace) -> None:
    if args.run_id:
        base = Path("logs/probe_runs") / args.run_id
        base.mkdir(parents=True, exist_ok=True)
        if args.dataset_log == str(DEFAULT_DATASET_LOG):
            args.dataset_log = str(base / "affect_data.jsonl")
        if args.summary_log == str(DEFAULT_SUMMARY_LOG):
            args.summary_log = str(base / "probe_affect_log.txt")
        if args.state_file == str(DEFAULT_STATE_FILE):
            args.state_file = str(base / "affect_data_state.json")
    scenarios = _load_scenarios(args.scenario_file)
    if not scenarios:
        raise SystemExit("No scenarios available for the affect data probe.")
    scenario_prompt_counts = [len(s["prompts"]) for s in scenarios]
    state_path = Path(args.state_file)
    state = _load_state(state_path, args.profiles, len(scenarios), scenario_prompt_counts)
    dataset_log = Path(args.dataset_log)
    summary_log = Path(args.summary_log)
    cooldown = max(0.0, args.cooldown_seconds)
    target_turns = max(0, args.target_turns)
    iteration_limit = target_turns if target_turns else max(0, args.max_iterations)
    initial_total = int(state.get("iteration", 0))
    completed = 0

    with _memory_guard():
        try:
            while not iteration_limit or completed < iteration_limit:
                for profile in args.profiles:
                    if iteration_limit and completed >= iteration_limit:
                        break
                    settings_file = cp.PROFILE_FILES[profile]
                    pointers = state["profiles"][profile]
                    scenario_idx = pointers["scenario_index"]
                    prompt_idx = pointers["prompt_index"]
                    scenario = scenarios[scenario_idx]
                    prompt = scenario["prompts"][prompt_idx]
                    timestamp = cp._now_iso()
                    try:
                        payload = await _send_prompt(prompt, settings_file, args.reset_session)
                    except Exception as exc:  # noqa: BLE001
                        error_line = (
                            f"[{timestamp}] profile={profile} scenario={scenario['name']} "
                            f"prompt={prompt_idx + 1}/{len(scenario['prompts'])} status=error:{exc}"
                        )
                        _append_summary(summary_log, payload=error_line)
                        raise
                    completed += 1
                    state["iteration"] = initial_total + completed
                    dataset_row = _build_dataset_row(
                        timestamp=timestamp,
                        profile=profile,
                        scenario=scenario,
                        prompt_index=prompt_idx,
                        total_prompts=len(scenario["prompts"]),
                        prompt=prompt,
                        payload=payload,
                        iteration=state["iteration"],
                    )
                    _append_dataset_row(dataset_log, dataset_row)
                    next_iso = _next_timestamp(cooldown) if (not iteration_limit or completed < iteration_limit) else "n/a"
                    summary_line = (
                        f"[{timestamp}] profile={profile} scenario={scenario['name']} "
                        f"prompt={prompt_idx + 1}/{len(scenario['prompts'])} iteration={state['iteration']} "
                        f"status=ok next={next_iso}"
                    )
                    _append_summary(summary_log, payload=summary_line)
                    _advance_pointer(pointers, len(scenarios), len(scenario["prompts"]))
                    _save_state(state_path, state)
                    if cooldown > 0 and (not iteration_limit or completed < iteration_limit):
                        await cp._cooldown_sleep(cooldown)
                else:
                    continue
                break
        except KeyboardInterrupt:
            print("\n[affect_data_probe] Interrupted by user; state saved for resume.")
        finally:
            _save_state(state_path, state)
    if iteration_limit and completed >= iteration_limit:
        print("[affect_data_probe] Reached max iterations; exiting.")


def _build_dataset_row(
    *,
    timestamp: str,
    profile: str,
    scenario: Mapping[str, Any],
    prompt_index: int,
    total_prompts: int,
    prompt: str,
    payload: Mapping[str, Any],
    iteration: int,
) -> dict[str, Any]:
    state_block = payload.get("state") or {}
    entry = {
        "timestamp": timestamp,
        "profile": profile,
        "scenario": scenario.get("name"),
        "stimulus": scenario.get("stimulus"),
        "tags": scenario.get("tags") or [],
        "prompt_index": prompt_index,
        "prompt_total": total_prompts,
        "text": prompt,
        "targets": scenario.get("targets") or {},
        "reply": payload.get("reply"),
        "source": payload.get("source"),
        "length_plan": (payload.get("length_plan") or {}).get("label"),
        "reinforcement": payload.get("reinforcement"),
        "controller": payload.get("controller"),
        "hormones": (state_block.get("hormones") if isinstance(state_block, Mapping) else None),
        "mood": (state_block.get("mood") if isinstance(state_block, Mapping) else None),
        "iteration": iteration,
    }
    return entry


def _entrypoint() -> None:
    args = parse_args()
    asyncio.run(_run_probe(args))


if __name__ == "__main__":
    _entrypoint()
