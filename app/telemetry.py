"""Telemetry and logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from state_engine import StateEngine

from app.config import current_profile
from app.constants import METRIC_THRESHOLDS
from app.runtime import RuntimeState


def log_json_line(
    path: Path,
    payload: Mapping[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Append a JSON payload to the given log path."""
    if not payload:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        if logger:
            logger.debug("Failed to append json line to %s: %s", path, exc)


def log_sampling_snapshot(
    snapshot: Mapping[str, Any],
    path: Path,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Persist the latest sampling snapshot for offline review."""
    if not snapshot:
        return
    log_json_line(path, snapshot, logger=logger)


def log_reinforcement_metrics(
    payload: Mapping[str, Any],
    path: Path,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Append reinforcement metrics for offline calibration."""
    log_json_line(path, payload, logger=logger)


def log_endocrine_turn(
    payload: Mapping[str, Any],
    path: Path,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Persist detailed per-turn endocrine diagnostics for offline modelling."""
    log_json_line(path, payload, logger=logger)


def log_affect_classification(
    user_text: str,
    classification: Any | None,
    path: Path,
    *,
    shorten: Callable[[str, int], str],
    logger: logging.Logger | None = None,
) -> None:
    """Persist affect classifier outputs for offline calibration."""
    if not classification:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "text": shorten(user_text, 160),
        "classification": classification.as_dict(),
    }
    log_json_line(path, payload, logger=logger)


def log_hormone_trace_event(
    trace: Mapping[str, Any] | None,
    *,
    telemetry: Mapping[str, Any] | None,
    reinforcement: Mapping[str, Any] | None,
    user: str,
    reply: str,
    intent: str,
    length_label: str | None,
    path: Path,
    enabled: bool,
    shorten: Callable[[str, int], str],
    logger: logging.Logger | None = None,
) -> None:
    """Capture verbose hormone clamp traces when enabled."""
    if not enabled or not trace:
        return
    profile = telemetry.get("profile") if telemetry else current_profile()
    sampling = telemetry.get("sampling") if telemetry else None
    hormone_sampling = telemetry.get("hormone_sampling") if telemetry else None
    policy_preview = telemetry.get("policy_preview") if telemetry else None
    controller_snapshot = telemetry.get("controller") if telemetry else None
    controller_input = telemetry.get("controller_input") if telemetry else None
    pre_snapshot = telemetry.get("pre") if telemetry else None
    engine = telemetry.get("engine") if telemetry else None
    model_alias = telemetry.get("model_alias") if telemetry else None
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "profile": profile,
        "intent": intent,
        "length_label": length_label,
        "engine": engine,
        "model_alias": model_alias,
        "user": shorten(user, 200),
        "reply": shorten(reply, 220),
        "trace": dict(trace),
    }
    if sampling:
        payload["sampling"] = dict(sampling)
    if hormone_sampling:
        payload["hormone_sampling"] = dict(hormone_sampling)
    if policy_preview:
        payload["policy_preview"] = dict(policy_preview)
    if controller_snapshot:
        payload["controller"] = controller_snapshot
    if controller_input:
        payload["controller_input"] = controller_input
    if pre_snapshot:
        payload["pre"] = pre_snapshot
    if reinforcement:
        payload["reinforcement"] = dict(reinforcement)
    log_json_line(path, payload, logger=logger)


def log_voice_guard_event(
    verdict: Mapping[str, Any] | None,
    *,
    user: str,
    reply: str,
    intent: str,
    profile: str,
    path: Path,
    shorten: Callable[[str, int], str],
    logger: logging.Logger | None = None,
) -> None:
    """Append a voice guard verdict."""
    if not verdict:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": shorten(user, 200),
        "reply": shorten(reply, 220),
        "intent": intent,
        "profile": profile,
        "verdict": verdict,
    }
    log_json_line(path, payload, logger=logger)


def log_webui_interaction(
    *,
    user: str,
    reply: str,
    intent: str,
    profile: str,
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
    path: Path,
    pretty_path: Path,
    shorten: Callable[[str, int], str],
    logger: logging.Logger | None = None,
) -> None:
    """Log interactions for the operator web UI."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": shorten(user, 240),
        "reply": shorten(reply, 280),
        "intent": intent,
        "profile": profile,
        "voice_guard": voice_guard or {},
    }
    if telemetry:
        payload["sampling"] = telemetry.get("sampling")
        payload["controller"] = telemetry.get("controller")
        payload["hormones"] = (telemetry.get("pre") or {}).get("hormones")
    log_json_line(path, payload, logger=logger)
    log_webui_interaction_pretty(payload, pretty_path, shorten=shorten, logger=logger)


def log_webui_interaction_pretty(
    payload: Mapping[str, Any],
    path: Path,
    *,
    shorten: Callable[[str, int], str],
    logger: logging.Logger | None = None,
) -> None:
    """Append a human-friendly snapshot of the latest interaction."""
    if not payload:
        return
    voice_guard = payload.get("voice_guard") or {}
    voice_flag = "flagged" if voice_guard.get("flagged") else "clear"
    voice_score = voice_guard.get("score", 0.0)
    sampling = payload.get("sampling") or {}
    controller = payload.get("controller") or {}
    hormones = payload.get("hormones") or {}
    sampling_line = ""
    if sampling:
        sampling_line = (
            f"temp={sampling.get('temperature')}, top_p={sampling.get('top_p')}, "
            f"max_tokens={sampling.get('max_tokens')}"
        )
        if sampling.get("self_bias_scale") is not None:
            sampling_line += f", self_bias={sampling.get('self_bias_scale')}"
    controller_line = ""
    if controller:
        adjustments = controller.get("adjustments") or {}
        max_delta = adjustments.get("max_tokens_delta")
        controller_line = (
            f"controller.max_tokens={controller.get('applied', {}).get('max_tokens')} "
            f"(delta={max_delta})"
        )
    hormone_line = ""
    if hormones:
        dopamine = hormones.get("dopamine")
        oxytocin = hormones.get("oxytocin")
        cortisol = hormones.get("cortisol")
        hormone_line = (
            f"dopamine={dopamine}, oxytocin={oxytocin}, cortisol={cortisol}"
        )
    pretty = "\n".join(
        [
            f"{payload.get('timestamp', '')} | intent={payload.get('intent', '')} | profile={payload.get('profile', '')}",
            f"user: {shorten(str(payload.get('user') or ''), 240)}",
            f"reply: {shorten(str(payload.get('reply') or ''), 280)}",
            f"voice_guard={voice_flag}({voice_score})",
            sampling_line,
            controller_line,
            hormone_line,
            "",
        ]
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(pretty)
    except Exception as exc:  # pragma: no cover - diagnostics only
        if logger:
            logger.debug("Failed to write readable web UI log: %s", exc)


def compose_turn_telemetry(
    *,
    context: Mapping[str, Any],
    sampling: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    state_engine: StateEngine,
    shorten: Callable[[str, int], str],
    model_alias: str,
) -> dict[str, Any]:
    """Produce the per-turn telemetry payload used by logging and responses."""
    memory_block = context.get("memory") or {}
    long_term_preview = [shorten(str(record), 160) for record in memory_block.get("long_term", [])]
    working_entries = memory_block.get("working", [])
    pre_snapshot = {
        "mood": context.get("mood"),
        "hormones": context.get("hormones"),
        "traits": list(state_engine.trait_tags()),
        "memory_summary": shorten(memory_block.get("summary", ""), 160),
        "working": list(working_entries) if isinstance(working_entries, list) else [],
        "long_term_preview": long_term_preview,
    }
    affect_context = context.get("affect") or state_engine.affect_overview()
    if affect_context:
        traits_map = affect_context.get("traits")
        if isinstance(traits_map, dict):
            pre_snapshot["trait_overview"] = {key: round(float(value), 4) for key, value in traits_map.items()}
        tags = affect_context.get("tags")
        if isinstance(tags, (list, tuple)):
            pre_snapshot["affect_tags"] = list(tags)
    telemetry = {
        "timestamp": snapshot.get("timestamp"),
        "profile": snapshot.get("profile", current_profile()),
        "model_alias": model_alias,
        "sampling": dict(sampling),
        "policy_preview": context.get("sampling_policy_preview"),
        "pre": pre_snapshot,
    }
    hormone_sampling = snapshot.get("hormone_sampling")
    if hormone_sampling:
        telemetry["hormone_sampling"] = hormone_sampling
    controller_snapshot = snapshot.get("controller")
    if controller_snapshot:
        telemetry["controller"] = controller_snapshot
    controller_input = snapshot.get("controller_input")
    if controller_input:
        telemetry["controller_input"] = controller_input
    style_hits = snapshot.get("hormone_style_hits")
    if style_hits:
        telemetry["hormone_style_hits"] = style_hits
    affect_overrides = snapshot.get("affect_overrides")
    if affect_overrides:
        telemetry["affect_overrides"] = affect_overrides
    user_affect_snapshot = snapshot.get("user_affect")
    if user_affect_snapshot and "user_affect" not in telemetry:
        telemetry["user_affect"] = user_affect_snapshot
    helper_penalty = snapshot.get("helper_penalty")
    if helper_penalty:
        telemetry["helper_penalty"] = helper_penalty
    return telemetry


def _compute_metric_delta(value: float | None, threshold: float, mode: str) -> float | None:
    if value is None:
        return None
    if mode == "min":
        return value - threshold
    return threshold - value


def _compute_metric_progress(value: float | None, threshold: float, mode: str) -> float | None:
    if value is None:
        return None
    if threshold <= 0:
        return None
    if mode == "min":
        return max(0.0, min(1.0, value / threshold))
    if value <= 0:
        return None
    return max(0.0, min(1.0, threshold / value))


def _metric_history_lists(runtime_state: RuntimeState) -> dict[str, list[float]]:
    return {
        "authenticity_score": list(runtime_state.auth_history),
        "assistant_drift": list(runtime_state.drift_history),
        "self_preoccupation": list(runtime_state.self_history),
        "affect_valence": list(runtime_state.affect_valence_history),
        "affect_intimacy": list(runtime_state.affect_intimacy_history),
        "affect_tension": list(runtime_state.affect_tension_history),
    }


def compose_live_status(
    *,
    state_engine: StateEngine,
    runtime_state: RuntimeState,
    model_alias: str,
    local_llama_available: bool,
) -> dict[str, Any]:
    """Compile telemetry data for dashboards and watchdogs."""
    state_payload = state_engine.get_state()
    metric_histories = _metric_history_lists(runtime_state)
    metric_summary: dict[str, Any] = {}
    for name, (threshold, mode) in METRIC_THRESHOLDS.items():
        value = runtime_state.last_metric_averages.get(name)
        meets_target: bool | None = None
        if value is not None:
            meets_target = value >= threshold if mode == "min" else value <= threshold
        metric_summary[name] = {
            "value": value,
            "threshold": threshold,
            "mode": mode,
            "delta": _compute_metric_delta(value, threshold, mode),
            "progress": _compute_metric_progress(value, threshold, mode),
            "meets_target": meets_target,
            "recent": metric_histories.get(name, []),
        }
    metric_summary["samples_seen"] = runtime_state.metric_sample_counter
    metric_summary["last_reinforcement"] = dict(runtime_state.last_reinforcement_metrics)
    affect_metrics = {
        "affect_valence": {
            "value": runtime_state.last_metric_averages.get("affect_valence"),
            "recent": metric_histories.get("affect_valence", []),
        },
        "affect_intimacy": {
            "value": runtime_state.last_metric_averages.get("affect_intimacy"),
            "recent": metric_histories.get("affect_intimacy", []),
        },
        "affect_tension": {
            "value": runtime_state.last_metric_averages.get("affect_tension"),
            "recent": metric_histories.get("affect_tension", []),
        },
    }
    affect_overview = state_engine.affect_overview()
    traits_snapshot = state_engine.trait_snapshot()
    traits_payload: dict[str, Any] | None = None
    if traits_snapshot:
        traits_payload = {
            "steadiness": round(traits_snapshot.steadiness, 4),
            "curiosity": round(traits_snapshot.curiosity, 4),
            "warmth": round(traits_snapshot.warmth, 4),
            "tension": round(traits_snapshot.tension, 4),
        }
    controller_payload: dict[str, Any] | None = None
    if runtime_state.last_controller_applied:
        controller_payload = dict(runtime_state.last_controller_applied)
    controller_input: dict[str, Any] | None = None
    if runtime_state.last_controller_features:
        controller_input = {
            "features": {
                key: round(float(value), 6) for key, value in runtime_state.last_controller_features.items()
            },
            "tags": list(runtime_state.last_controller_tags),
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "session_id": runtime_state.session_counter,
        "profile": current_profile(),
        "model_alias": model_alias,
        "local_engine": local_llama_available,
        "state": state_payload,
        "metrics": metric_summary,
        "affect_metrics": affect_metrics,
        "affect": affect_overview,
        "traits": traits_payload,
        "controller": controller_payload,
        "controller_input": controller_input,
        "last_hormone_delta": dict(runtime_state.last_hormone_delta or {}),
    }


def write_telemetry_snapshot(
    telemetry: Mapping[str, Any] | None,
    path: Path,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Persist the latest telemetry blob to disk for dashboards."""
    if not telemetry:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - diagnostics only
        if logger:
            logger.debug("Failed to write telemetry snapshot: %s", exc)


__all__ = [
    "log_affect_classification",
    "log_endocrine_turn",
    "log_hormone_trace_event",
    "log_json_line",
    "log_reinforcement_metrics",
    "log_sampling_snapshot",
    "log_voice_guard_event",
    "log_webui_interaction",
    "log_webui_interaction_pretty",
    "compose_live_status",
    "compose_turn_telemetry",
    "write_telemetry_snapshot",
]
