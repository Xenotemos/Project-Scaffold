"""Controller coordination helpers extracted from the API entrypoint."""

from __future__ import annotations

import asyncio
from datetime import datetime
from threading import Lock
from typing import Any, Awaitable, Callable, Mapping, Sequence

from brain.controller_policy import ControllerPolicyRuntime, ControllerStepResult
from state_engine import StateEngine, TraitSnapshot

from app.constants import (
    CONTROLLER_HORMONE_SCALE,
    CONTROLLER_MAX_TAGS,
    HELPER_TONE_WORDS,
    LOW_SELF_RELEASE_STREAK,
    LOW_SELF_RELEASE_THRESHOLD,
    METRIC_THRESHOLDS,
    OUTWARD_ATTENTION_WORDS,
    OUTWARD_RELEASE_FLOOR,
    RECOVERY_RELEASE_STREAK,
    SELF_ATTENUATION_WORDS,
    SELF_OBSERVATION_WORDS,
)
from app.runtime import RuntimeState

ResetSessionFn = Callable[[str], Awaitable[None]]


def _normalize_hormone_feature(name: str, value: float, baseline: Mapping[str, float]) -> float:
    base = float(baseline.get(name, 50.0))
    normalized = (float(value) - base) / CONTROLLER_HORMONE_SCALE
    return max(-1.5, min(1.5, normalized))


def gather_active_tags(state_engine: StateEngine, limit: int = CONTROLLER_MAX_TAGS) -> list[str]:
    """Combine memory and trait tags for controller consumption."""
    tags: list[str] = []
    memory_tags = state_engine.memory_manager.active_tags(limit=limit)
    for tag in memory_tags:
        lowered = str(tag).lower()
        if lowered not in tags:
            tags.append(lowered)
            if len(tags) >= limit:
                return tags[:limit]
    for tag in state_engine.trait_tags():
        lowered = str(tag).lower()
        if lowered not in tags:
            tags.append(lowered)
            if len(tags) >= limit:
                break
    return tags[:limit]


def build_controller_feature_map(
    *,
    state_engine: StateEngine,
    runtime_state: RuntimeState,
    traits: TraitSnapshot | None,
    hormones: Mapping[str, float] | None,
    intent: str,
    length_label: str | None,
    profile: str,
    tags: Sequence[str],
) -> dict[str, float]:
    """Encode controller features from current internal state."""
    features: dict[str, float] = {"bias": 1.0}
    traits = traits or TraitSnapshot(steadiness=0.0, curiosity=0.0, warmth=0.0, tension=0.0)
    features["trait:steadiness"] = float(traits.steadiness)
    features["trait:curiosity"] = float(traits.curiosity)
    features["trait:warmth"] = float(traits.warmth)
    features["trait:tension"] = float(traits.tension)

    hormone_state = hormones or {}
    baseline = state_engine.hormone_system.baseline()
    for name in ("dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"):
        value = float(hormone_state.get(name, baseline.get(name, 50.0)))
        features[f"hormone:{name}"] = _normalize_hormone_feature(name, value, baseline)

    intent_key = (intent or "").strip().lower()
    if intent_key:
        features[f"intent:{intent_key}"] = 1.0

    length_key = (length_label or "").strip().lower()
    if length_key:
        features[f"length:{length_key}"] = 1.0

    profile_key = (profile or "").strip().lower()
    if profile_key:
        features[f"profile:{profile_key}"] = 1.0

    # Surface tag presence directly as binary features.
    for tag in tags:
        label = str(tag).strip().lower()
        if label:
            features[f"tag:{label}"] = 1.0
    metrics_payload: Mapping[str, float] | None = None
    if isinstance(runtime_state.last_reinforcement_metrics, dict):
        latest_metrics = runtime_state.last_reinforcement_metrics.get("metrics")
        if isinstance(latest_metrics, dict):
            metrics_payload = latest_metrics
        else:
            metrics_payload = runtime_state.last_reinforcement_metrics
    if metrics_payload:
        for name in ("authenticity_score", "self_preoccupation", "assistant_drift", "outward_streak_score"):
            value = metrics_payload.get(name)
            if isinstance(value, (int, float)):
                features[f"metric:{name}"] = float(value)
    return features


def run_controller_policy(
    controller_runtime: ControllerPolicyRuntime | None,
    controller_lock: Lock,
    runtime_state: RuntimeState,
    feature_map: Mapping[str, float],
    tags: Sequence[str],
) -> ControllerStepResult | None:
    """Evaluate the controller policy with the supplied features."""
    if controller_runtime is None:
        runtime_state.last_controller_result = None
        runtime_state.last_controller_features = None
        runtime_state.last_controller_applied = None
        runtime_state.last_controller_tags = ()
        return None
    with controller_lock:
        result = controller_runtime.step(feature_map, tags=tags)
        runtime_state.last_controller_result = result
        runtime_state.last_controller_features = dict(feature_map)
        runtime_state.last_controller_applied = None
        runtime_state.last_controller_tags = tuple(str(tag) for tag in tags)
        return result


def apply_controller_adjustments(
    runtime_state: RuntimeState,
    sampling: dict[str, Any],
    adjustments: Mapping[str, float],
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
    max_completion_tokens: int,
    min_tokens_floor: int | None = None,
    reset_session: ResetSessionFn | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blend controller adjustments into the sampling package."""
    if not adjustments:
        return sampling, {}
    updated = dict(sampling)
    applied: dict[str, Any] = {}

    def _round(value: float) -> float:
        return round(float(value), 4)

    temp_delta = float(adjustments.get("temperature_delta", 0.0))
    if abs(temp_delta) > 1e-5:
        base_temp = float(updated.get("temperature", base_temperature))
        new_temp = max(0.3, min(1.45, base_temp + temp_delta))
        updated["temperature"] = _round(new_temp)
        applied["temperature"] = updated["temperature"]

    top_p_delta = float(adjustments.get("top_p_delta", 0.0))
    if abs(top_p_delta) > 1e-5:
        base_top = float(updated.get("top_p", base_top_p))
        new_top = max(0.35, min(0.995, base_top + top_p_delta))
        updated["top_p"] = _round(new_top)
        applied["top_p"] = updated["top_p"]

    freq_delta = float(adjustments.get("frequency_penalty_delta", 0.0))
    if abs(freq_delta) > 1e-5:
        base_freq = float(updated.get("frequency_penalty", base_frequency_penalty))
        new_freq = max(-0.5, min(1.8, base_freq + freq_delta))
        updated["frequency_penalty"] = _round(new_freq)
        applied["frequency_penalty"] = updated["frequency_penalty"]

    presence_delta = float(adjustments.get("presence_penalty_delta", 0.0))
    if abs(presence_delta) > 1e-5:
        base_presence = float(updated.get("presence_penalty", 0.1))
        new_presence = max(-0.5, min(1.8, base_presence + presence_delta))
        updated["presence_penalty"] = _round(new_presence)
        applied["presence_penalty"] = updated["presence_penalty"]

    metrics_payload: dict[str, float] = {}
    if isinstance(runtime_state.last_reinforcement_metrics, dict):
        latest_metrics = runtime_state.last_reinforcement_metrics.get("metrics")
        if isinstance(latest_metrics, dict):
            metrics_payload = latest_metrics
        else:
            metrics_payload = runtime_state.last_reinforcement_metrics  # older format compatibility

    floor_tokens = None
    if isinstance(min_tokens_floor, (int, float)):
        floor_tokens = max(64, int(min_tokens_floor))

    def _enforce_token_floor(value: int) -> int:
        if floor_tokens is not None and value < floor_tokens:
            return floor_tokens
        return value

    tokens_delta = float(adjustments.get("max_tokens_delta", 0.0))
    if tokens_delta < 0.0:
        auth_now = metrics_payload.get("authenticity_score")
        self_focus_now = metrics_payload.get("self_preoccupation")
        if isinstance(auth_now, (int, float)) and isinstance(self_focus_now, (int, float)):
            if auth_now >= 0.42 and self_focus_now <= 0.6:
                tokens_delta = max(tokens_delta, -120.0)
        if floor_tokens is not None:
            base_tokens = int(updated.get("max_tokens", max_completion_tokens))
            min_delta = floor_tokens - base_tokens
            if min_delta > tokens_delta:
                tokens_delta = min_delta
    if abs(tokens_delta) > 1e-3:
        base_tokens = int(updated.get("max_tokens", max_completion_tokens))
        new_tokens = max(64, min(1024, int(round(base_tokens + tokens_delta))))
        new_tokens = _enforce_token_floor(new_tokens)
        updated["max_tokens"] = new_tokens
        applied["max_tokens"] = new_tokens

    bias_scale = float(adjustments.get("self_bias_scale", 0.0))
    auth_now = float(metrics_payload.get("authenticity_score", 0.0) or 0.0)
    auth_trend = float(runtime_state.last_metric_averages.get("authenticity", 0.0) or 0.0)
    auth_threshold = METRIC_THRESHOLDS.get("authenticity_score", (0.45, "min"))[0]
    auth_ok = isinstance(auth_now, (int, float)) and auth_now >= auth_threshold
    outward_value = float(metrics_payload.get("outward_streak_score", 0.0) or 0.0)
    outward_now = outward_value
    outward_bonus = outward_value >= OUTWARD_RELEASE_FLOOR
    strong_outward = outward_value >= 0.3
    readings = [
        value
        for value in (
            metrics_payload.get("self_preoccupation"),
            runtime_state.last_metric_averages.get("self_preoccupation"),
        )
        if isinstance(value, (int, float))
    ]
    high_self = max(readings) if readings else None
    average_self = float(runtime_state.last_metric_averages.get("self_preoccupation", 0.0) or 0.0)
    peak_candidates = [value for value in (high_self, average_self) if isinstance(value, (int, float))]
    peak_self = max(peak_candidates) if peak_candidates else None
    outward_scale = 0.0
    inversion_scale = 0.0
    clamp_severity = 0.0
    clear_self_bias = False
    clamp_triggered = False

    if peak_self is not None and peak_self >= 0.74:
        runtime_state.self_focus_streak += 1
    else:
        runtime_state.self_focus_streak = 0

    latest_self = metrics_payload.get("self_preoccupation")
    low_self_now = isinstance(latest_self, (int, float)) and latest_self <= LOW_SELF_RELEASE_THRESHOLD
    low_self_relax = low_self_now and runtime_state.recovery_lowself_streak >= LOW_SELF_RELEASE_STREAK
    recovery_active = runtime_state.clamp_recovery_turns > 0

    auth_deficit = max(0.0, 0.42 - auth_now)
    trend_deficit = max(0.0, 0.44 - auth_trend)
    if (peak_self is None or peak_self < 0.66) and not recovery_active:
        if auth_deficit or trend_deficit:
            boost = min(0.45, (auth_deficit * 1.35) + (trend_deficit * 1.05))
            if boost > 0.0:
                desired_floor = min(0.42, 0.18 + boost)
                if bias_scale < desired_floor:
                    bias_scale = desired_floor
                outward_scale = max(outward_scale, 0.06 + boost * 0.45)

    if high_self is not None and high_self > 0.64:
        overshoot = high_self - 0.64
        clamp = min(0.95, overshoot * 2.4)
        bias_scale *= max(0.05, 1.0 - clamp)
        outward_scale = max(outward_scale, clamp * 0.75)
        if high_self > 0.82:
            inversion_scale = max(inversion_scale, min(0.6, (high_self - 0.82) * 3.0))
    if average_self > 0.7:
        trend_penalty = min(0.6, (average_self - 0.7) * 2.0)
        bias_scale *= max(0.05, 1.0 - trend_penalty)
        outward_scale = max(outward_scale, trend_penalty * 0.5)
        if average_self > 0.8:
            inversion_scale = max(inversion_scale, min(0.45, (average_self - 0.8) * 2.5))

    if peak_self is not None and peak_self >= 0.68:
        clamp_severity = min(
            1.0,
            max(0.0, peak_self - 0.68) * 2.2 + max(0.0, average_self - 0.66) * 1.4,
        )
        clear_self_bias = True
        bias_scale *= max(0.05, 1.0 - clamp_severity * 1.05)
        if bias_scale > 0.0:
            bias_scale = max(0.0, bias_scale - clamp_severity * 0.45)
        outward_scale = max(outward_scale, 0.32 + clamp_severity * 0.55)
        inversion_scale = max(inversion_scale, clamp_severity * 0.6)
        clamp_triggered = True
        runtime_state.clamp_recovery_turns = max(runtime_state.clamp_recovery_turns, 4)
        runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 3)

    hard_clamp = runtime_state.self_focus_streak >= 2
    if hard_clamp:
        clear_self_bias = True
        bias_scale = min(bias_scale, -0.35)
        outward_scale = max(outward_scale, 0.55)
        inversion_scale = max(inversion_scale, 0.75)
        clamp_triggered = True
        runtime_state.clamp_recovery_turns = max(runtime_state.clamp_recovery_turns, 5)
        runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 4)

    if clamp_severity >= 0.25:
        bias_scale = min(bias_scale, -(0.12 + clamp_severity * 0.5))
        outward_scale = max(outward_scale, 0.4 + clamp_severity * 0.4)
        clamp_triggered = True
        runtime_state.clamp_recovery_turns = max(runtime_state.clamp_recovery_turns, 4)
        runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 3)

    recovery_active = runtime_state.clamp_recovery_turns > 0
    if recovery_active and not clamp_triggered:
        clear_self_bias = True
        damp = max(0.3, min(0.6, 0.1 * runtime_state.clamp_recovery_turns))
        bias_scale = min(bias_scale, -damp)
        outward_scale = max(outward_scale, 0.4 + damp)
        inversion_scale = max(inversion_scale, 0.55 + (0.05 * runtime_state.clamp_recovery_turns))

    bias_scale = max(-0.45, min(0.5, bias_scale))
    updated["self_bias_scale"] = round(bias_scale, 4)

    existing_bias = dict(updated.get("logit_bias_words") or {})
    applied_bias: dict[str, float] = {}
    if clear_self_bias and existing_bias:
        for word in list(existing_bias.keys()):
            if word.lower() in SELF_OBSERVATION_WORDS:
                existing_bias[word] = 0.0

    if abs(bias_scale) > 1e-5:
        for word, weight in SELF_OBSERVATION_WORDS.items():
            bias_value = round(weight * bias_scale, 3)
            if abs(bias_value) < 1e-4:
                continue
            accumulated = round(existing_bias.get(word, 0.0) + bias_value, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated
    if inversion_scale > 1e-5:
        for word, weight in SELF_OBSERVATION_WORDS.items():
            bias_value = round(-weight * inversion_scale, 3)
            if abs(bias_value) < 1e-4:
                continue
            accumulated = round(existing_bias.get(word, 0.0) + bias_value, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated
    if outward_scale > 1e-5:
        for word, weight in OUTWARD_ATTENTION_WORDS.items():
            bias_value = round(weight * outward_scale, 3)
            if abs(bias_value) < 1e-4:
                continue
            accumulated = round(existing_bias.get(word, 0.0) + bias_value, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated

    priming_active = runtime_state.clamp_priming_turns > 0

    def _maybe_decay_recovery_window() -> None:
        nonlocal recovery_active
        if not recovery_active:
            runtime_state.recovery_good_streak = 0
            runtime_state.recovery_lowself_streak = 0
            return
        applied["recovery_window"] = runtime_state.clamp_recovery_turns
        if clamp_triggered or runtime_state.clamp_recovery_turns <= 0:
            return
        threshold = METRIC_THRESHOLDS.get("self_preoccupation", (0.75, "max"))[0]
        latest_value = latest_self if isinstance(latest_self, (int, float)) else None
        low_self_release = latest_value is not None and latest_value <= LOW_SELF_RELEASE_THRESHOLD
        improving = (
            latest_value is not None
            and isinstance(runtime_state.last_metric_averages.get("self_preoccupation"), (int, float))
            and latest_value < runtime_state.last_metric_averages.get("self_preoccupation", 1.0)
        )
        if low_self_release or improving:
            runtime_state.recovery_good_streak = min(
                runtime_state.recovery_good_streak + 1,
                RECOVERY_RELEASE_STREAK,
            )
        else:
            runtime_state.recovery_good_streak = 0
        if low_self_release:
            runtime_state.recovery_lowself_streak += 1
        else:
            runtime_state.recovery_lowself_streak = 0
        if (
            runtime_state.recovery_good_streak >= RECOVERY_RELEASE_STREAK
            or runtime_state.recovery_lowself_streak >= LOW_SELF_RELEASE_STREAK
        ):
            runtime_state.clamp_recovery_turns = max(0, runtime_state.clamp_recovery_turns - 1)
            runtime_state.recovery_good_streak = max(0, runtime_state.recovery_good_streak - RECOVERY_RELEASE_STREAK)
            runtime_state.recovery_lowself_streak = max(0, runtime_state.recovery_lowself_streak - LOW_SELF_RELEASE_STREAK)
            recovery_active = runtime_state.clamp_recovery_turns > 0

    priming_spike = 0.0
    if recovery_active or clamp_triggered or priming_active or runtime_state.reset_priming_bias > 1e-4:
        priming_spike = max(0.0, runtime_state.reset_priming_bias)
        streak_decay = min(runtime_state.recovery_good_streak, 3)
        decay_factor = max(0.45, 1.0 - 0.2 * streak_decay)
        damp_strength = 0.5 + (0.08 * max(runtime_state.clamp_recovery_turns, 1)) + (0.05 * runtime_state.clamp_priming_turns)
        damp_strength *= decay_factor
        if auth_ok or strong_outward:
            damp_strength *= 0.7
        elif outward_bonus:
            damp_strength *= 0.8
        elif low_self_now:
            damp_strength *= 0.85
        damp_strength = min(0.9, max(0.25, damp_strength))
        for word, weight in SELF_ATTENUATION_WORDS.items():
            bias_value = round(-weight * damp_strength, 3)
            accumulated = round(existing_bias.get(word, 0.0) + bias_value, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated
        outward_floor = max(0.4, 0.25 + 0.08 * (runtime_state.clamp_recovery_turns + runtime_state.clamp_priming_turns))
        for word, weight in OUTWARD_ATTENTION_WORDS.items():
            outward_boost = round(weight * (outward_floor + priming_spike), 3)
            accumulated = round(existing_bias.get(word, 0.0) + outward_boost, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated
        if priming_active or priming_spike > 1e-4:
            applied["priming_trace"] = {
                "auth": round(auth_now, 4),
                "self": round(float(latest_self), 4) if isinstance(latest_self, (int, float)) else None,
                "bias_spike": round(priming_spike, 3) if priming_spike > 1e-4 else 0.0,
                "outward": round(float(outward_now), 4) if isinstance(outward_now, (int, float)) else None,
            }
        if priming_spike > 1e-4:
            runtime_state.reset_priming_bias = 0.0
    helper_penalty_scale = max(0.0, runtime_state.helper_drift_level)
    if helper_penalty_scale > 1e-4:
        for word, weight in HELPER_TONE_WORDS.items():
            penalty = round(-abs(weight) * helper_penalty_scale, 3)
            if abs(penalty) < 1e-4:
                continue
            accumulated = round(existing_bias.get(word, 0.0) + penalty, 3)
            existing_bias[word] = accumulated
            applied_bias[word] = accumulated
        applied["helper_penalty_scale"] = round(helper_penalty_scale, 3)
    if applied_bias:
        updated["logit_bias_words"] = existing_bias
        applied["logit_bias_words"] = applied_bias
    if abs(bias_scale) > 1e-5:
        applied["self_bias_scale"] = round(bias_scale, 4)
    if outward_scale > 1e-5:
        applied["outward_bias_scale"] = round(outward_scale, 4)
    if inversion_scale > 1e-5:
        applied["self_bias_inversion"] = round(inversion_scale, 4)
    if priming_active:
        applied["priming_window"] = runtime_state.clamp_priming_turns
        runtime_state.clamp_priming_turns = max(0, runtime_state.clamp_priming_turns - 1)

    if clamp_severity > 1e-4 or hard_clamp or recovery_active:
        current_temp = float(updated.get("temperature", base_temperature))
        temp_drop = min(0.38, 0.18 + clamp_severity * 0.3 + (0.12 if hard_clamp else 0.0))
        new_temp = max(0.25, current_temp - temp_drop)
        if new_temp != current_temp:
            updated["temperature"] = _round(new_temp)
            applied["temperature"] = updated["temperature"]
        current_top = float(updated.get("top_p", base_top_p))
        top_drop = min(0.3, 0.16 + clamp_severity * 0.28 + (0.1 if hard_clamp else 0.0))
        new_top = max(0.35, current_top - top_drop)
        if new_top != current_top:
            updated["top_p"] = _round(new_top)
            applied["top_p"] = updated["top_p"]
        freq_bump = min(0.65, 0.22 + clamp_severity * 0.5 + (0.12 if hard_clamp else 0.0))
        base_freq = float(updated.get("frequency_penalty", base_frequency_penalty))
        new_freq = max(-0.5, min(1.8, base_freq + freq_bump))
        if new_freq != base_freq:
            updated["frequency_penalty"] = _round(new_freq)
            applied["frequency_penalty"] = updated["frequency_penalty"]
        current_tokens = int(updated.get("max_tokens", max_completion_tokens))
        token_trim = min(
            420,
            int(round(520 * clamp_severity + (180 if hard_clamp else 0))),
        )
        new_tokens = max(64, current_tokens - token_trim)
        new_tokens = _enforce_token_floor(new_tokens)
        if new_tokens != current_tokens:
            updated["max_tokens"] = new_tokens
            applied["max_tokens"] = new_tokens

    if low_self_relax:
        relaxed_tokens = min(max_completion_tokens, max(int(updated.get("max_tokens", 0) or 0), 160))
        relaxed_tokens = _enforce_token_floor(relaxed_tokens)
        if relaxed_tokens and relaxed_tokens != updated.get("max_tokens"):
            updated["max_tokens"] = relaxed_tokens
            applied["max_tokens"] = relaxed_tokens
        relaxed_temp = max(float(updated.get("temperature", base_temperature)), min(base_temperature, 0.68))
        if relaxed_temp != updated.get("temperature"):
            updated["temperature"] = _round(relaxed_temp)
            applied["temperature"] = updated["temperature"]
        relaxed_top = max(float(updated.get("top_p", base_top_p)), 0.8)
        if relaxed_top != updated.get("top_p"):
            updated["top_p"] = _round(relaxed_top)
            applied["top_p"] = updated["top_p"]

    if outward_scale > 1e-5:
        updated["outward_bias_scale"] = round(outward_scale, 4)
    if inversion_scale > 1e-5:
        updated["self_bias_inversion"] = round(inversion_scale, 4)

    _maybe_decay_recovery_window()

    runtime_state.last_controller_applied = dict(applied) if applied else {}
    if clamp_triggered:
        runtime_state.recovery_good_streak = 0
        runtime_state.recovery_lowself_streak = 0
        now = datetime.now().astimezone()
        if runtime_state.last_clamp_reset and (now - runtime_state.last_clamp_reset).total_seconds() <= 300:
            applied["session_reset"] = "skipped_recent"
        else:
            if reset_session is None:
                applied["session_reset"] = "skipped_missing_callback"
            else:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    applied["session_reset"] = "skipped_no_loop"
                else:
                    loop.create_task(reset_session("controller_clamp"))
                    runtime_state.last_clamp_reset = now
                    applied["session_reset"] = "queued"

    return updated, applied


def controller_trace_snapshot(runtime_state: RuntimeState) -> dict[str, Any] | None:
    """Return the latest controller evaluation for logging/memory coupling."""
    if runtime_state.last_controller_result is None:
        return None
    trace: dict[str, Any] = {
        "adjustments": {
            key: round(float(value), 6) for key, value in runtime_state.last_controller_result.adjustments.items()
        },
        "raw_outputs": [round(float(value), 6) for value in runtime_state.last_controller_result.raw_outputs],
        "hidden_state": [round(float(value), 6) for value in runtime_state.last_controller_result.hidden_state],
    }
    if runtime_state.last_controller_applied:
        applied_payload: dict[str, Any] = {}
        for key, value in runtime_state.last_controller_applied.items():
            if isinstance(value, (int, float)):
                applied_payload[key] = value if isinstance(value, int) else round(float(value), 6)
            else:
                applied_payload[key] = value
        trace["applied"] = applied_payload
    if runtime_state.last_controller_features:
        trace["features"] = {
            key: round(float(value), 6) for key, value in runtime_state.last_controller_features.items()
        }
    return trace


__all__ = [
    "apply_controller_adjustments",
    "build_controller_feature_map",
    "controller_trace_snapshot",
    "gather_active_tags",
    "run_controller_policy",
]
