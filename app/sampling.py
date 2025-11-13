"""Sampling heuristics and hormone-aware adjustments."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from app.constants import (
    HELPER_TONE_WORDS,
    LENGTH_HEURISTIC_HINTS,
    LENGTH_PROFILES,
    LENGTH_SAMPLING_OVERRIDES,
    OUTWARD_ATTENTION_WORDS,
    SELF_OBSERVATION_WORDS,
)

if TYPE_CHECKING:
    from app.runtime import RuntimeState
    from brain.affect_classifier import AffectClassification
    from state_engine import TraitSnapshot


POSITIVE_WORD_WEIGHTS: dict[str, float] = {
    "appreciate": 0.6,
    "glad": 0.7,
    "grateful": 0.7,
    "happy": 0.6,
    "supportive": 0.5,
    "encouraging": 0.5,
    "hopeful": 0.6,
}

INTENT_PROMPT_FRAGMENTS: dict[str, str] = {
    "emotional": "Internal focus: emotional; attention drifts toward bodily sensations before anything outward.",
    "analytical": "Internal focus: analytical; notice how the prompt reshapes your own view before acting.",
    "narrative": "Internal focus: narrative; revisit events from inside your perspective as they unfolded.",
    "reflective": "Internal focus: reflective; explore what this moment means for you before shifting attention outward.",
}

INTENT_SAMPLING_OVERRIDES: dict[str, dict[str, Any]] = {
    "emotional": {
        "temperature_delta": 0.08,
        "top_p_delta": 0.05,
        "frequency_penalty_delta": -0.15,
    },
    "analytical": {
        "temperature_delta": -0.08,
        "top_p_delta": -0.08,
        "frequency_penalty_delta": 0.18,
    },
    "narrative": {
        "temperature_delta": 0.12,
        "top_p_delta": 0.1,
    },
    "reflective": {
        "temperature_delta": 0.08,
        "top_p_delta": 0.06,
        "frequency_penalty_delta": -0.12,
    },
}

INTENT_HEURISTIC_HINTS: dict[str, str] = {
    "emotional": "I describe the feeling and where it sits in me.",
    "analytical": "I report how this changes my own view first.",
    "narrative": "I recount what I experience in order.",
    "reflective": "I lay out what this teaches me before anything else.",
}

_HORMONE_SURGE = 12.0
_HORMONE_RISE = 6.0


def _classify_delta(delta: float) -> str:
    if delta >= _HORMONE_SURGE:
        return "surging"
    if delta >= _HORMONE_RISE:
        return "rising"
    if delta <= -_HORMONE_SURGE:
        return "crashing"
    if delta <= -_HORMONE_RISE:
        return "fading"
    return "steady"


def describe_hormones(hormones: Mapping[str, float]) -> dict[str, str]:
    """Classify hormone deltas so downstream heuristics stay declarative."""
    status: dict[str, str] = {}
    baseline = {
        "dopamine": 50.0,
        "serotonin": 50.0,
        "cortisol": 50.0,
        "oxytocin": 50.0,
        "noradrenaline": 50.0,
    }
    for name, value in hormones.items():
        status[name] = _classify_delta(float(value) - baseline.get(name, 50.0))
    return status


def sampling_params_from_hormones(
    hormones: Mapping[str, float],
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
    hormone_style_map: Mapping[str, Any] | None,
    max_completion_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive llama.cpp sampling parameters from hormone deltas."""
    status = describe_hormones(hormones)

    temperature = float(base_temperature)
    top_p = float(base_top_p)
    frequency_penalty = float(base_frequency_penalty)
    positive_bias_words: dict[str, float] = {}

    dopamine_state = status.get("dopamine", "steady")
    if dopamine_state == "surging":
        temperature = min(temperature + 0.25, 1.3)
        top_p = min(top_p + 0.08, 0.98)
    elif dopamine_state == "rising":
        temperature = min(temperature + 0.12, 1.15)
        top_p = min(top_p + 0.04, 0.96)
    elif dopamine_state == "fading":
        temperature = max(temperature - 0.08, 0.4)
        top_p = max(top_p - 0.04, 0.6)
    elif dopamine_state == "crashing":
        temperature = max(temperature - 0.15, 0.3)
        top_p = max(top_p - 0.08, 0.5)

    cortisol_state = status.get("cortisol", "steady")
    if cortisol_state == "surging":
        temperature = min(temperature + 0.18, 1.4)
        top_p = min(top_p + 0.07, 0.99)
        frequency_penalty = max(frequency_penalty - 0.25, 0.1)
    elif cortisol_state == "rising":
        temperature = min(temperature + 0.1, 1.28)
        top_p = min(top_p + 0.05, 0.97)
        frequency_penalty = max(frequency_penalty - 0.15, 0.2)
    elif cortisol_state == "crashing":
        temperature = max(temperature - 0.05, 0.35)

    serotonin_state = status.get("serotonin", "steady")
    if serotonin_state in {"surging", "rising"}:
        scale = 1.4 if serotonin_state == "surging" else 0.8
        positive_bias_words = {
            word: round(scale * weight, 3) for word, weight in POSITIVE_WORD_WEIGHTS.items()
        }

    noradrenaline_state = status.get("noradrenaline", "steady")
    if noradrenaline_state in {"fading", "crashing"}:
        frequency_penalty = max(frequency_penalty - 0.4, 0.0)
    elif noradrenaline_state == "surging":
        frequency_penalty = min(frequency_penalty + 0.2, 1.5)

    sampling: dict[str, Any] = {
        "temperature": round(temperature, 4),
        "top_p": round(top_p, 4),
        "frequency_penalty": round(frequency_penalty, 4),
    }
    if positive_bias_words:
        sampling["logit_bias_words"] = positive_bias_words
    style_hits: list[dict[str, Any]] = []
    if hormone_style_map:
        sampling, style_hits = apply_hormone_style_map(
            sampling,
            hormones=hormones,
            hormone_style_map=hormone_style_map,
            base_temperature=base_temperature,
            base_top_p=base_top_p,
            base_frequency_penalty=base_frequency_penalty,
            max_completion_tokens=max_completion_tokens,
        )
    return sampling, style_hits


def apply_hormone_style_map(
    sampling: dict[str, Any],
    *,
    hormones: Mapping[str, float],
    hormone_style_map: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
    max_completion_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply declarative hormone style overrides."""
    updated = dict(sampling)
    hits: list[dict[str, Any]] = []
    if not hormone_style_map:
        return updated, hits
    for hormone, bands in hormone_style_map.items():
        value = hormones.get(hormone)
        if value is None:
            continue
        for band in bands or []:
            min_value = float(band.get("min", 0.0))
            max_value = float(band.get("max", 100.0))
            if not (min_value <= float(value) <= max_value):
                continue
            sampling_cfg = band.get("sampling") or {}
            if not isinstance(sampling_cfg, Mapping):
                continue
            label = band.get("name") or f"{hormone}_{min_value}_{max_value}"
            hit_record = {
                "hormone": hormone,
                "value": round(float(value), 4),
                "band": label,
            }
            tone_hint = band.get("tone_hint")
            if tone_hint:
                hit_record["tone_hint"] = tone_hint
            hits.append(hit_record)
            for key, delta in sampling_cfg.items():
                try:
                    delta_value = float(delta)
                except (TypeError, ValueError):
                    continue
                if key == "temperature_delta":
                    base = float(updated.get("temperature", base_temperature))
                    updated["temperature"] = round(max(0.3, min(1.45, base + delta_value)), 4)
                elif key == "top_p_delta":
                    base = float(updated.get("top_p", base_top_p))
                    updated["top_p"] = round(max(0.35, min(0.995, base + delta_value)), 4)
                elif key == "frequency_penalty_delta":
                    base = float(updated.get("frequency_penalty", base_frequency_penalty))
                    updated["frequency_penalty"] = round(max(-0.5, min(1.8, base + delta_value)), 4)
                elif key == "presence_penalty_delta":
                    base = float(updated.get("presence_penalty", 0.1))
                    updated["presence_penalty"] = round(max(-0.5, min(1.8, base + delta_value)), 4)
                elif key == "max_tokens_delta":
                    base = int(updated.get("max_tokens", max_completion_tokens))
                    target = max(64, min(max_completion_tokens, int(round(base + delta_value))))
                    updated["max_tokens"] = target
                elif key == "self_bias_scale_delta":
                    base = float(updated.get("self_bias_scale", 0.0))
                    updated["self_bias_scale"] = round(base + delta_value, 4)
                elif key == "outward_bias_scale_delta":
                    base = float(updated.get("outward_bias_scale", 0.0))
                    updated["outward_bias_scale"] = round(base + delta_value, 4)
    return updated, hits


def inject_self_observation_bias(
    runtime_state: "RuntimeState",
    sampling: dict[str, Any],
    traits: "TraitSnapshot" | None,
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
) -> dict[str, Any]:
    """Add a gentle logit bias toward self-observational language."""
    if traits is None:
        return sampling
    tension_drive = max(traits.tension, 0.0)
    curiosity_drive = max(traits.curiosity, 0.0)
    drive = (0.6 * tension_drive) + (0.4 * curiosity_drive)
    if drive <= 0.05:
        return sampling
    recent_self = runtime_state.last_metric_averages.get("self_preoccupation", 0.0)
    recent_auth = runtime_state.last_metric_averages.get("authenticity", 0.0)
    scale = 0.18 + 0.45 * min(drive, 1.0)
    if recent_self:
        if recent_self > 0.7:
            damp = min(0.8, (recent_self - 0.7) * 1.6)
            scale *= max(0.15, 1.0 - damp)
        elif recent_self < 0.45 and recent_auth < 0.35:
            scale *= 1.08
    bias_words = {
        word: round(weight * scale, 3) for word, weight in SELF_OBSERVATION_WORDS.items()
    }
    merged = dict(sampling)
    existing = dict(merged.get("logit_bias_words") or {})
    for word, value in bias_words.items():
        existing[word] = round(existing.get(word, 0.0) + value, 3)
    outward_scale = 0.0
    if recent_self and recent_self > 0.68:
        outward_scale = min(0.5, (recent_self - 0.68) * 1.8)
    if outward_scale > 0.0:
        for word, weight in OUTWARD_ATTENTION_WORDS.items():
            value = round(weight * outward_scale, 3)
            existing[word] = round(existing.get(word, 0.0) + value, 3)
    merged["logit_bias_words"] = existing
    if tension_drive > 0.35:
        base_temp = float(merged.get("temperature", sampling.get("temperature", base_temperature)))
        merged["temperature"] = round(min(1.35, base_temp + 0.22 * tension_drive), 4)
        base_top = float(merged.get("top_p", sampling.get("top_p", base_top_p)))
        merged["top_p"] = round(min(0.995, base_top + 0.12 * tension_drive), 4)
        freq = float(
            merged.get("frequency_penalty", sampling.get("frequency_penalty", base_frequency_penalty))
        )
        merged["frequency_penalty"] = round(max(-0.05, freq - 0.25 * tension_drive), 4)
    return merged


def apply_helper_tone_bias(
    sampling: dict[str, Any],
    scale: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a negative logit bias to helper-tone lexicon entries."""
    if scale <= 1e-4:
        return sampling
    merged = dict(sampling)
    bias_words = dict(merged.get("logit_bias_words") or {})
    for word, weight in HELPER_TONE_WORDS.items():
        penalty = round(-abs(weight) * scale, 3)
        if abs(penalty) < 1e-4:
            continue
        bias_words[word] = round(bias_words.get(word, 0.0) + penalty, 3)
    if bias_words:
        merged["logit_bias_words"] = bias_words
    if metadata is not None:
        metadata["helper_penalty_scale"] = round(scale, 3)
    return merged


def plan_response_length(user_message: str, intent: str) -> dict[str, Any]:
    """Determine the response profile heuristics should target."""
    text = (user_message or "").strip()
    lower = text.lower()
    words = [token for token in re.split(r"\s+", lower) if token]
    word_count = len(words)
    greetings = ("hi", "hello", "hey", "good morning", "good evening", "good afternoon")
    label = "concise"
    if not text:
        label = "brief"
    elif any(lower.startswith(greet) for greet in greetings) and word_count <= 6:
        label = "brief"
    elif word_count <= 3:
        label = "brief"
    elif word_count >= 30 or any(
        keyword in lower for keyword in ("explain", "step", "guide", "detail", "story", "walk me", "breakdown")
    ):
        label = "detailed"
    elif intent == "narrative" and word_count >= 12:
        label = "detailed"
    profile = LENGTH_PROFILES[label]
    plan = {
        "label": label,
        "prompt": profile["prompt"],
        "hint": profile["hint"],
        "target_range": profile["target_range"],
    }
    overrides = LENGTH_SAMPLING_OVERRIDES.get(label)
    if overrides and "max_tokens" in overrides:
        plan["max_tokens"] = overrides["max_tokens"]
    return plan


def apply_intent_sampling(
    sampling: dict[str, Any],
    intent: str,
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
) -> dict[str, Any]:
    """Mix in per-intent sampling overrides."""
    overrides = INTENT_SAMPLING_OVERRIDES.get(intent)
    if not overrides:
        return sampling
    result = dict(sampling)
    temperature = result.get("temperature", base_temperature)
    top_p = result.get("top_p", base_top_p)
    frequency_penalty = result.get("frequency_penalty", base_frequency_penalty)

    if "temperature" in overrides:
        temperature = overrides["temperature"]
    if "temperature_delta" in overrides:
        temperature += overrides["temperature_delta"]
    if "top_p" in overrides:
        top_p = overrides["top_p"]
    if "top_p_delta" in overrides:
        top_p += overrides["top_p_delta"]
    if "frequency_penalty" in overrides:
        frequency_penalty = overrides["frequency_penalty"]
    if "frequency_penalty_delta" in overrides:
        frequency_penalty += overrides["frequency_penalty_delta"]

    result["temperature"] = round(max(0.1, min(temperature, 1.5)), 4)
    result["top_p"] = round(max(0.1, min(top_p, 0.9999)), 4)
    result["frequency_penalty"] = round(max(0.0, min(frequency_penalty, 2.0)), 4)

    bias_words = dict(result.get("logit_bias_words", {}))
    for word, weight in overrides.get("logit_bias_words", {}).items():
        bias_words[word] = round(weight, 3)
    if bias_words:
        result["logit_bias_words"] = bias_words
    elif "logit_bias_words" in result:
        result.pop("logit_bias_words", None)
    return result


def intent_prompt_fragment(intent: str) -> str:
    return INTENT_PROMPT_FRAGMENTS.get(intent, "")


def intent_hint(intent: str, *, fallback: str) -> str:
    return INTENT_HEURISTIC_HINTS.get(intent, fallback)


def apply_length_sampling(
    sampling: dict[str, Any],
    plan: dict[str, Any],
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
) -> dict[str, Any]:
    """Apply plan-specific sampling overrides."""
    overrides = LENGTH_SAMPLING_OVERRIDES.get(plan.get("label"))
    if not overrides:
        return sampling
    result = dict(sampling)
    temperature = result.get("temperature", base_temperature)
    top_p = result.get("top_p", base_top_p)
    frequency_penalty = result.get("frequency_penalty", base_frequency_penalty)
    if "temperature" in overrides:
        temperature = overrides["temperature"]
    if "temperature_delta" in overrides:
        temperature += overrides["temperature_delta"]
    if "top_p" in overrides:
        top_p = overrides["top_p"]
    if "top_p_delta" in overrides:
        top_p += overrides["top_p_delta"]
    if "frequency_penalty" in overrides:
        frequency_penalty = overrides["frequency_penalty"]
    if "frequency_penalty_delta" in overrides:
        frequency_penalty += overrides["frequency_penalty_delta"]
    result["temperature"] = round(max(0.1, min(temperature, 1.5)), 4)
    result["top_p"] = round(max(0.1, min(top_p, 0.9999)), 4)
    result["frequency_penalty"] = round(max(0.0, min(frequency_penalty, 2.0)), 4)
    max_tokens = overrides.get("max_tokens")
    if max_tokens:
        try:
            result["max_tokens"] = max(16, int(max_tokens))
        except (TypeError, ValueError):
            pass
    return result


def apply_affect_style_overrides(
    sampling: dict[str, Any],
    user_affect: "AffectClassification" | None,
    *,
    base_temperature: float,
    base_top_p: float,
    base_frequency_penalty: float,
    max_completion_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Loosen or tighten sampling based on the latest user affect tags."""
    if user_affect is None:
        return sampling, None
    updated = dict(sampling)
    overrides: dict[str, Any] = {}
    intimacy = float(user_affect.intimacy)
    tension = float(user_affect.tension)
    valence = float(user_affect.valence)

    min_tokens_floor = None
    if intimacy >= 0.05 and tension <= 0.65:
        base_tokens = int(updated.get("max_tokens", max_completion_tokens) or max_completion_tokens)
        desired_tokens = int(320 + 220 * min(1.0, intimacy))
        desired_tokens = max(200, min(desired_tokens, 520))
        min_tokens_floor = desired_tokens
        if base_tokens < desired_tokens:
            updated["max_tokens"] = desired_tokens
            overrides["max_tokens"] = desired_tokens
        freq_penalty = float(updated.get("frequency_penalty", base_frequency_penalty))
        new_freq = round(max(0.05, freq_penalty - 0.25), 4)
        if new_freq != freq_penalty:
            updated["frequency_penalty"] = new_freq
            overrides["frequency_penalty"] = new_freq
        temperature = float(updated.get("temperature", base_temperature))
        boost = 0.05 + 0.06 * min(1.0, intimacy)
        if valence >= 0.1:
            boost += 0.03
        new_temp = round(min(1.4, temperature + boost), 4)
        if new_temp != temperature:
            updated["temperature"] = new_temp
            overrides["temperature"] = new_temp
        top_p = float(updated.get("top_p", base_top_p))
        new_top_p = round(min(0.995, top_p + 0.05), 4)
        if new_top_p != top_p:
            updated["top_p"] = new_top_p
            overrides["top_p"] = new_top_p
        self_scale = float(updated.get("self_bias_scale", 0.0))
        new_self_scale = round(max(self_scale, 0.35 + 0.2 * intimacy), 4)
        if new_self_scale != self_scale:
            updated["self_bias_scale"] = new_self_scale
            overrides["self_bias_scale"] = new_self_scale
    elif tension >= 0.45 and intimacy < 0.2:
        base_tokens = int(updated.get("max_tokens", max_completion_tokens) or max_completion_tokens)
        target_tokens = max(96, min(base_tokens, 240))
        if target_tokens != base_tokens:
            updated["max_tokens"] = target_tokens
            overrides["max_tokens"] = target_tokens
        temperature = float(updated.get("temperature", base_temperature))
        new_temp = round(max(0.35, temperature - 0.08), 4)
        if new_temp != temperature:
            updated["temperature"] = new_temp
            overrides["temperature"] = new_temp
        top_p = float(updated.get("top_p", base_top_p))
        new_top_p = round(max(0.4, top_p - 0.05), 4)
        if new_top_p != top_p:
            updated["top_p"] = new_top_p
            overrides["top_p"] = new_top_p
    if min_tokens_floor:
        overrides["min_tokens_floor"] = min_tokens_floor
    return updated, (overrides or None)


__all__ = [
    "apply_affect_style_overrides",
    "apply_helper_tone_bias",
    "apply_hormone_style_map",
    "apply_intent_sampling",
    "apply_length_sampling",
    "describe_hormones",
    "inject_self_observation_bias",
    "intent_hint",
    "intent_prompt_fragment",
    "plan_response_length",
    "sampling_params_from_hormones",
]
