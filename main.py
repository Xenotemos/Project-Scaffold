"""FastAPI entrypoint for the Living AI project."""

from __future__ import annotations

import asyncio
import copy
import logging
import json
import math
import os
import re
import shlex
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from brain.affect_classifier import (
    AffectClassification,
    AffectClassifier,
    load_affect_classifier,
)
from brain.affect_sidecar import AffectSidecarManager
from brain.controller_policy import ControllerPolicy, ControllerPolicyRuntime, load_controller_policy
from brain.llm_client import LivingLLMClient
from brain.local_llama_engine import LocalLlamaEngine
from brain.intent_router import IntentPrediction, predict_intent
from brain.policy import SamplingPolicy, derive_policy
from brain.reinforcement import reset_outward_streak, score_response
from brain.voice_guard import VoiceGuard
from state_engine import StateEngine, TraitSnapshot
from brain.hormone_model import HormoneDynamicsModel, load_model
from app.constants import (
    DEFAULT_HORMONE_STYLE_MAP,
    HELPER_PENALTY_DECAY,
    HELPER_PENALTY_STEP,
    LENGTH_HEURISTIC_HINTS,
    LENGTH_SAMPLING_OVERRIDES,
    LOW_SELF_BASE_EXTRA_TURNS,
    LOW_SELF_BASE_OUTWARD_FLOOR,
    LOW_SELF_BASE_PRIMING_MULTIPLIER,
    LOW_SELF_INSTRUCT_DRIFT_CEILING,
    LOW_SELF_INSTRUCT_OUTWARD_FLOOR,
    LOW_SELF_INSTRUCT_PRIMING_MULTIPLIER,
    LOW_SELF_RELEASE_STREAK,
    LOW_SELF_RELEASE_THRESHOLD,
    LOW_SELF_SUCCESS_AUTH,
    LOW_SELF_SUCCESS_DRIFT_CEILING,
    LOW_SELF_SUCCESS_MAX_SELF,
    LOW_SELF_SUCCESS_PRIMING,
    LOW_SELF_SUCCESS_STREAK_TARGET,
    MAX_HELPER_REGEN_ATTEMPTS,
    METRIC_THRESHOLDS,
    OUTWARD_RELEASE_FLOOR,
    RECOVERY_RELEASE_STREAK,
    RESET_PRIMING_BIAS_DEFAULT,
)
from app.settings import RuntimeSettings, clear_settings_cache
from app.runtime import RuntimeState
from app.chat_context import build_chat_context
from app.config import current_profile, current_settings_file, resolve_settings_file
from app.persona import (
    apply_persona_feedback,
    build_persona_snapshot,
    compose_heuristic_reply,
    collect_helper_sample,
    collect_persona_sample,
    extract_focus_phrase,
    record_internal_reflection,
    update_self_narration,
)
from app.telemetry import (
    compose_live_status,
    compose_turn_telemetry,
    log_affect_classification,
    log_endocrine_turn,
    log_hormone_trace_event,
    log_json_line,
    log_reinforcement_metrics,
    log_sampling_snapshot,
    write_telemetry_snapshot,
    log_voice_guard_event,
    log_webui_interaction,
)
from app.affect_telemetry import log_affect_head_event
from app.affect_tail import append_affect_compact, append_affect_raw
from app.sampling import (
    apply_affect_style_overrides,
    apply_helper_tone_bias,
    apply_intent_sampling,
    apply_length_sampling,
    describe_hormones,
    inject_self_observation_bias,
    intent_hint as sampling_intent_hint,
    intent_prompt_fragment,
    plan_response_length,
    sampling_params_from_hormones,
)
from app.controller import (
    apply_controller_adjustments,
    build_controller_feature_map,
    controller_trace_snapshot,
    gather_active_tags,
    run_controller_policy,
)

TELEMETRY_INTERVAL_SECONDS = float(os.getenv("TELEMETRY_REFRESH_SECONDS", "1.0"))

app = FastAPI(title="Living AI Project")
state_engine = StateEngine()
update_task: asyncio.Task[None] | None = None
telemetry_process: asyncio.subprocess.Process | None = None
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
logger = logging.getLogger("living_ai.main")
runtime_settings = RuntimeSettings.load()
runtime_state = RuntimeState()
SETTINGS: dict[str, Any] = dict(runtime_settings.raw)
BASE_TEMPERATURE = runtime_settings.base_temperature
BASE_TOP_P = runtime_settings.base_top_p
BASE_FREQUENCY_PENALTY = runtime_settings.base_frequency_penalty
LLM_ENDPOINT = runtime_settings.llm_endpoint
LLM_TIMEOUT = runtime_settings.llm_timeout
llm_client: LivingLLMClient | None = None
LLAMA_SERVER_BIN = runtime_settings.llama_server_bin
LLAMA_MODEL_PATH = runtime_settings.llama_model_path
LLAMA_SERVER_HOST = runtime_settings.llama_server_host
LLAMA_SERVER_PORT = runtime_settings.llama_server_port
LLAMA_MODEL_ALIAS = runtime_settings.llama_model_alias
LLAMA_COMPLETION_TOKENS = runtime_settings.llama_completion_tokens
LLAMA_SERVER_ARGS: list[str] = list(runtime_settings.llama_server_args)
LLAMA_READINESS_TIMEOUT = runtime_settings.llama_readiness_timeout
LLAMA_SERVER_TIMEOUT = runtime_settings.llama_server_timeout
AFFECT_CONTEXT_ENABLED = runtime_settings.affect_context_enabled
AFFECT_SAMPLING_PREVIEW_ENABLED = runtime_settings.affect_sampling_preview_enabled
AFFECT_MEMORY_PREVIEW_ENABLED = runtime_settings.affect_memory_preview_enabled
AFFECT_RECENCY_WINDOW_SECONDS = runtime_settings.affect_recency_window_seconds
AFFECT_SAMPLING_BLEND_WEIGHT = runtime_settings.affect_sampling_blend_weight
AFFECT_DEBUG_PANEL_ENABLED = runtime_settings.affect_debug_panel_enabled
AFFECT_LOG_ROOT = BASE_DIR / "docs" / "logs"
AFFECT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
WEBUI_LOG_DIR = AFFECT_LOG_ROOT / "webui"
WEBUI_LOG_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_LOG_DIR = AFFECT_LOG_ROOT / "telemetry"
TELEMETRY_LOG_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_DIR = BASE_DIR / "docs" / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLING_SNAPSHOT_LOG = BASE_DIR / "logs" / "sampling_snapshots.jsonl"
REINFORCEMENT_LOG = BASE_DIR / "logs" / "reinforcement_metrics.jsonl"
ENDOCRINE_LOG = BASE_DIR / "logs" / "endocrine_turns.jsonl"
HORMONE_TRACE_LOG = BASE_DIR / "logs" / "hormone_trace.jsonl"
HORMONE_TRACE_ENABLED = True
AFFECT_CLASSIFIER_LOG = BASE_DIR / "logs" / "affect_classifier.jsonl"
AFFECT_HEAD_TELEMETRY_LOG = BASE_DIR / "logs" / "affect_head_telemetry.jsonl"
AFFECT_ALIGNMENT_LOG = BASE_DIR / "logs" / "affect_head_alignment.jsonl"
WEBUI_INTERACTION_LOG = WEBUI_LOG_DIR / "interactions.log"
WEBUI_INTERACTION_PRETTY_LOG = WEBUI_LOG_DIR / "interactions_readable.log"
TELEMETRY_SNAPSHOT_PATH = TELEMETRY_LOG_DIR / "last_frame.json"
VOICE_GUARD_LOG = AFFECT_LOG_ROOT / "voice_guard.jsonl"
PERSONA_SAMPLE_LOG = CORPUS_DIR / "persona_samples.jsonl"
HELPER_SAMPLE_LOG = CORPUS_DIR / "helper_tone_samples.jsonl"
HORMONE_MODEL_PATH = runtime_settings.hormone_model_path
HORMONE_MODEL: HormoneDynamicsModel | None = None
CONTROLLER_POLICY_PATH = runtime_settings.controller_policy_path
CONTROLLER_POLICY: ControllerPolicy | None = None
CONTROLLER_RUNTIME: ControllerPolicyRuntime | None = None
CONTROLLER_LOCK = threading.Lock()
AFFECT_CLASSIFIER_PATH = runtime_settings.affect_classifier_path
AFFECT_CLASSIFIER: AffectClassifier | None = None
AFFECT_CLASSIFIER_BLEND_MIN_CONFIDENCE = 0.05
AFFECT_SIDECAR: AffectSidecarManager | None = None
affect_head_console: subprocess.Popen | None = None
VOICE_GUARD = VoiceGuard()
HORMONE_STYLE_MAP_PATH = runtime_settings.hormone_style_map_path
HORMONE_STYLE_MAP: dict[str, list[dict[str, Any]]] = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)
AFFECT_CONTEXT_MAX_TURNS = 8
AFFECT_CONTEXT_DECAY_K = float(os.getenv("AFFECT_CONTEXT_DECAY_K", "0.35"))
AFFECT_CONTEXT_SNIPPET_CHARS = int(os.getenv("AFFECT_CONTEXT_SNIPPET_CHARS", "220"))


def _refresh_settings() -> None:
    """Reload project settings from disk and environment."""
    global runtime_settings
    global SETTINGS, BASE_TEMPERATURE, BASE_TOP_P, BASE_FREQUENCY_PENALTY
    global LLM_ENDPOINT, LLM_TIMEOUT
    global LLAMA_SERVER_BIN, LLAMA_MODEL_PATH, LLAMA_SERVER_HOST, LLAMA_SERVER_PORT
    global LLAMA_MODEL_ALIAS, LLAMA_COMPLETION_TOKENS, LLAMA_SERVER_ARGS
    global LLAMA_READINESS_TIMEOUT, LLAMA_SERVER_TIMEOUT
    global AFFECT_CONTEXT_ENABLED, AFFECT_SAMPLING_PREVIEW_ENABLED, AFFECT_MEMORY_PREVIEW_ENABLED
    global AFFECT_RECENCY_WINDOW_SECONDS, AFFECT_SAMPLING_BLEND_WEIGHT, AFFECT_DEBUG_PANEL_ENABLED
    global HORMONE_MODEL_PATH
    global CONTROLLER_POLICY_PATH
    global AFFECT_CLASSIFIER_PATH
    global HORMONE_STYLE_MAP_PATH
    global HORMONE_TRACE_ENABLED

    runtime_settings = RuntimeSettings.load()
    SETTINGS = dict(runtime_settings.raw)
    BASE_TEMPERATURE = runtime_settings.base_temperature
    BASE_TOP_P = runtime_settings.base_top_p
    BASE_FREQUENCY_PENALTY = runtime_settings.base_frequency_penalty
    LLM_ENDPOINT = runtime_settings.llm_endpoint
    LLM_TIMEOUT = runtime_settings.llm_timeout
    LLAMA_SERVER_BIN = runtime_settings.llama_server_bin
    LLAMA_MODEL_PATH = runtime_settings.llama_model_path
    LLAMA_SERVER_HOST = runtime_settings.llama_server_host
    LLAMA_SERVER_PORT = runtime_settings.llama_server_port
    LLAMA_MODEL_ALIAS = runtime_settings.llama_model_alias
    LLAMA_COMPLETION_TOKENS = runtime_settings.llama_completion_tokens
    LLAMA_SERVER_ARGS = list(runtime_settings.llama_server_args)
    LLAMA_READINESS_TIMEOUT = runtime_settings.llama_readiness_timeout
    LLAMA_SERVER_TIMEOUT = runtime_settings.llama_server_timeout
    HORMONE_MODEL_PATH = runtime_settings.hormone_model_path
    CONTROLLER_POLICY_PATH = runtime_settings.controller_policy_path
    AFFECT_CLASSIFIER_PATH = runtime_settings.affect_classifier_path
    HORMONE_STYLE_MAP_PATH = runtime_settings.hormone_style_map_path
    HORMONE_TRACE_ENABLED = runtime_settings.hormone_trace_enabled
    AFFECT_CONTEXT_ENABLED = runtime_settings.affect_context_enabled
    AFFECT_SAMPLING_PREVIEW_ENABLED = runtime_settings.affect_sampling_preview_enabled
    AFFECT_MEMORY_PREVIEW_ENABLED = runtime_settings.affect_memory_preview_enabled
    AFFECT_RECENCY_WINDOW_SECONDS = runtime_settings.affect_recency_window_seconds
    AFFECT_SAMPLING_BLEND_WEIGHT = runtime_settings.affect_sampling_blend_weight
    AFFECT_DEBUG_PANEL_ENABLED = runtime_settings.affect_debug_panel_enabled
    logger.info("Hormone tracing: %s", "enabled" if HORMONE_TRACE_ENABLED else "disabled")
    _reinitialize_hormone_model()
    _reinitialize_controller_policy()
    _reinitialize_affect_classifier()
    _reinitialize_hormone_style_map()


def _reinitialize_hormone_model() -> None:
    """Load the hormone dynamics model if configured."""
    global HORMONE_MODEL
    if not HORMONE_MODEL_PATH:
        HORMONE_MODEL = None
        return
    model_path = Path(HORMONE_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    model = load_model(model_path)
    if model is None:
        logger.warning("Failed to load hormone model from %s", model_path)
    HORMONE_MODEL = model


def _reinitialize_controller_policy() -> None:
    """Load the controller policy and reset its runtime state."""
    global CONTROLLER_POLICY, CONTROLLER_RUNTIME
    if not CONTROLLER_POLICY_PATH:
        CONTROLLER_POLICY = None
        CONTROLLER_RUNTIME = None
        runtime_state.last_controller_result = None
        return
    policy_path = Path(CONTROLLER_POLICY_PATH)
    if not policy_path.is_absolute():
        policy_path = BASE_DIR / policy_path
    policy = load_controller_policy(policy_path)
    if policy is None:
        logger.warning("Failed to load controller policy from %s", policy_path)
        CONTROLLER_POLICY = None
        CONTROLLER_RUNTIME = None
        runtime_state.last_controller_result = None
        return
    CONTROLLER_POLICY = policy
    CONTROLLER_RUNTIME = policy.runtime()
    runtime_state.last_controller_result = None


def _reinitialize_affect_classifier() -> None:
    """Load the affect classifier configuration."""
    global AFFECT_CLASSIFIER, AFFECT_SIDECAR
    if os.getenv("AFFECT_SIDECAR_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        AFFECT_CLASSIFIER = AffectClassifier()
        AFFECT_SIDECAR = None
        logger.info("Affect sidecar disabled via AFFECT_SIDECAR_DISABLE; using heuristic classifier.")
        return
    try:
        AFFECT_CLASSIFIER = load_affect_classifier(AFFECT_CLASSIFIER_PATH or None)
        # If the classifier uses a sidecar manager, store it for startup hooks
        if hasattr(AFFECT_CLASSIFIER, "_sidecar"):
            AFFECT_SIDECAR = getattr(AFFECT_CLASSIFIER, "_sidecar", None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load affect classifier config: %s", exc)
        AFFECT_CLASSIFIER = AffectClassifier()
        AFFECT_SIDECAR = None


def _reinitialize_hormone_style_map() -> None:
    """Load hormone style overrides from configuration."""
    global HORMONE_STYLE_MAP
    if not HORMONE_STYLE_MAP_PATH:
        HORMONE_STYLE_MAP = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)
        return
    path = Path(HORMONE_STYLE_MAP_PATH)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        HORMONE_STYLE_MAP = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            HORMONE_STYLE_MAP = data
        else:
            HORMONE_STYLE_MAP = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse hormone style map %s: %s", path, exc)
        HORMONE_STYLE_MAP = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)


_refresh_settings()
local_llama_engine: LocalLlamaEngine | None = None


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0]
    return shortened + "..."



def _apply_reinforcement_signals(
    signals: dict[str, float],
    *,
    length_plan: dict[str, Any] | None = None,
    reply_text: str | None = None,
    profile: str | None = None,
) -> dict[str, Any] | None:
    """Map reinforcement signals into hormone adjustments and surface diagnostics."""
    if not signals:
        return None
    valence = signals.get("valence_delta", 0.0)
    length_score = signals.get("length_score", 1.0)
    engagement = signals.get("engagement_score", 0.0)
    authenticity = signals.get("authenticity_score", 0.0)
    assistant_drift = signals.get("assistant_drift", 0.0)
    self_focus = signals.get("self_preoccupation", 0.0)
    affect_valence = signals.get("affect_valence", 0.0)
    affect_intimacy = signals.get("affect_intimacy", 0.0)
    affect_tension = signals.get("affect_tension", 0.0)
    input_valence = signals.get("input_affect_valence")
    input_intimacy = signals.get("input_affect_intimacy")
    input_tension = signals.get("input_affect_tension")
    input_safety = signals.get("input_affect_safety")
    input_arousal = signals.get("input_affect_arousal")
    input_approach = signals.get("input_affect_approach_avoid")
    input_inhib_social = signals.get("input_affect_inhibition_social")
    input_inhib_vuln = signals.get("input_affect_inhibition_vulnerability")
    input_inhib_self = signals.get("input_affect_inhibition_self_restraint")
    input_expectedness = signals.get("input_affect_expectedness")
    input_momentum = signals.get("input_affect_momentum_delta")
    input_intents = signals.get("input_affect_intent") or []
    input_rpe = signals.get("input_affect_rpe")
    classifier_conf = float(signals.get("affect_classifier_confidence", 0.0) or 0.0)
    affect_tags = {
        str(tag).lower()
        for tag in (signals.get("affect_classifier_tags") or [])
        if isinstance(tag, str)
    }
    has_input_affect = any(
        isinstance(value, (int, float)) for value in (input_valence, input_intimacy, input_tension)
    )

    def _classifier_weight(conf: float) -> float:
        if conf >= 0.65:
            return 0.9
        if conf >= 0.45:
            return 0.75
        if conf >= 0.25:
            return 0.6
        return 0.45

    weight = _classifier_weight(classifier_conf)

    def _blend_component(base_value: float, classified_value: float | None) -> float:
        if isinstance(classified_value, (int, float)):
            blended = ((1.0 - weight) * float(base_value)) + (weight * float(classified_value))
            return blended
        return float(base_value)

    adjusted_valence = _blend_component(float(valence), float(input_valence) if isinstance(input_valence, (int, float)) else None)
    valence_signal = adjusted_valence if has_input_affect else float(valence)
    affect_valence = float(max(-1.0, min(1.0, adjusted_valence)))

    raw_intimacy = _blend_component(float(affect_intimacy), float(input_intimacy) if isinstance(input_intimacy, (int, float)) else None)
    raw_tension = _blend_component(float(affect_tension), float(input_tension) if isinstance(input_tension, (int, float)) else None)
    intimacy_signal = max(-1.5, min(1.5, raw_intimacy if has_input_affect else float(affect_intimacy)))
    tension_signal = max(-1.5, min(1.5, raw_tension if has_input_affect else float(affect_tension)))

    adjustments: dict[str, float] = {}
    requested_adjustments: dict[str, float] = {}
    applied_adjustments: dict[str, float] = {}
    clamped_adjustments: dict[str, float] = {}
    pre_hormones = state_engine.hormone_system.get_state()
    post_hormones = dict(pre_hormones)

    if valence_signal > 0.05:
        lift = min(1.0, valence_signal)
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.0 * lift
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.9 * lift
        dopamine_gain = 0.8
        if profile == "instruct":
            dopamine_gain = 1.05
        elif profile == "base":
            dopamine_gain = 1.1
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + dopamine_gain * lift
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.6 * lift
    elif valence_signal < -0.05:
        drop = min(1.0, abs(valence_signal))
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.1 * drop
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.7 * drop
    if has_input_affect and affect_valence > 0.15:
        positivity = affect_valence - 0.15
        if profile == "instruct":
            dopamine_gain = 1.4
        elif profile == "base":
            dopamine_gain = 1.3
        else:
            dopamine_gain = 1.0
        if "tension" in affect_tags and tension_signal >= 0.25:
            dopamine_gain *= 0.4
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + dopamine_gain * positivity
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.9 * positivity
    elif has_input_affect and affect_valence < -0.15:
        negativity = abs(affect_valence) - 0.15
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.9 * negativity
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.6 * negativity
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.9 * negativity

    if length_score < 0.7:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 2.0
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 1.0
    elif length_score > 1.6:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 1.0
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.6

    if engagement < 0.35:
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 1.8
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.6
    elif engagement > 0.75:
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) - 1.0
    if intimacy_signal > 0.02:
        closeness = min(3.0, intimacy_signal * 4.5)
        if "tension" in affect_tags and tension_signal >= 0.25:
            closeness *= 0.35
        if profile == "instruct":
            dopamine_bonus = 1.55
        elif profile == "base":
            dopamine_bonus = 1.45
        else:
            dopamine_bonus = 1.0
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 2.2 * closeness
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + dopamine_bonus * closeness
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.55 * closeness
    if tension_signal > 0.02:
        spike = min(2.0, tension_signal * 3.2)
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.4 * spike
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 1.2 * spike
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.55 * spike
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.4 * spike
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.8 * spike
    elif tension_signal < -0.04:
        ease = min(1.5, abs(tension_signal) * 2.3)
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 1.05 * ease
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.85 * ease
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.55 * ease

    # Safety / arousal / approach-avoid
    if isinstance(input_safety, (int, float)):
        safety_mag = float(input_safety)
        if safety_mag < -0.2:
            pressure = abs(safety_mag)
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.1 * pressure
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.8 * pressure
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.7 * pressure
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.4 * pressure
        elif safety_mag > 0.2:
            ease = safety_mag
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 1.0 * ease
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.6 * ease
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.7 * ease
    if isinstance(input_arousal, (int, float)):
        arousal_mag = max(-1.0, min(1.0, float(input_arousal)))
        if arousal_mag > 0.15:
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.9 * arousal_mag
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.35 * arousal_mag
        elif arousal_mag < -0.15:
            ease = abs(arousal_mag)
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) - 0.7 * ease
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.4 * ease
    if isinstance(input_approach, (int, float)):
        approach = max(-1.0, min(1.0, float(input_approach)))
        if approach > 0.1:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.9 * approach
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.6 * approach
        elif approach < -0.1:
            avoidance = abs(approach)
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.8 * avoidance
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.6 * avoidance

    # Inhibition brakes damp dopamine/oxytocin and raise tension slightly
    for key, value, damp_factor in [
        ("inhibition_social", input_inhib_social, 0.5),
        ("inhibition_vulnerability", input_inhib_vuln, 0.6),
        ("inhibition_self_restraint", input_inhib_self, 0.7),
    ]:
        if isinstance(value, (int, float)) and value > 0.35:
            extra = (float(value) - 0.35) * damp_factor
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.6 * extra
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.5 * extra
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.4 * extra

    if isinstance(input_rpe, (int, float)):
        rpe_val = max(-1.0, min(1.0, float(input_rpe)))
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.8 * rpe_val
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.3 * rpe_val

    # Expectedness / momentum spikes
    if input_expectedness in {"mild_surprise", "strong_surprise"}:
        spike = 0.45 if input_expectedness == "mild_surprise" else 0.9
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + spike
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.5 * spike
    if input_momentum in {"soft_turn", "hard_turn"}:
        spike = 0.35 if input_momentum == "soft_turn" else 0.7
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + spike

    # Intent-driven tweaks
    for intent_label in input_intents:
        intent = str(intent_label).strip().lower()
        if intent in {"reassure", "comfort"}:
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.6
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.4
        elif intent in {"flirt_playful", "intimate"}:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.7
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.5
        elif intent in {"boundary"}:
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.6
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.4
        elif intent in {"manipulate", "dominate"}:
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.7
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.5
        elif intent in {"apologize"}:
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.35
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.35
        elif intent in {"vent"}:
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.4
        elif intent in {"seek_support"}:
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.45


    if authenticity >= 0.45:
        auth_bonus = max(0.0, authenticity - 0.45)
        focus_penalty = max(0.0, self_focus - 0.62)
        damp_factor = max(0.2, 1.0 - (1.4 * focus_penalty))
        dopamine_lift = (0.85 + 0.7 * min(auth_bonus, 0.4)) * damp_factor
        oxytocin_lift = 0.55 * damp_factor
        cortisol_drop = 0.45 * damp_factor
        if dopamine_lift:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + dopamine_lift
        if oxytocin_lift:
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + oxytocin_lift
        if cortisol_drop:
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - cortisol_drop
        if focus_penalty > 0.0:
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.5 * focus_penalty
    elif authenticity < 0.25:
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.8
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.5
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.4

    if assistant_drift >= 0.45:
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 1.1
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.7
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.4
    if assistant_drift >= 0.75:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.9

    if self_focus > 0.66:
        excess = (self_focus - 0.66) * 3.0
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.9 * excess
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.75 * excess
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.45 * excess
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.05 * excess
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.35 * excess
        if self_focus > 0.82:
            spike = (self_focus - 0.82) * 4.0
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.6 * spike
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.9 * spike
    elif self_focus < 0.35:
        lift = (0.35 - self_focus) * 1.8
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.4 * lift
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.3 * lift
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) - 0.3 * lift

    if length_plan and reply_text:
        target_range = length_plan.get("target_range")
        if isinstance(target_range, (list, tuple)) and len(target_range) == 2:
            min_target, max_target = target_range
        else:
            min_target, max_target = 1, 3
        sentence_count = _count_sentences(reply_text)
        if sentence_count and sentence_count < min_target:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.8
            adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 0.6
        elif sentence_count and sentence_count > max_target:
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 1.0
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.7

    if adjustments:
        state_engine.hormone_system.apply_deltas(adjustments)
        post_hormones = state_engine.hormone_system.get_state()
        requested_adjustments = {
            name: round(change, 4) for name, change in adjustments.items() if abs(change) >= 0.0001
        }
        for name, after in post_hormones.items():
            before = pre_hormones.get(name, 0.0)
            delta = round(after - before, 4)
            if abs(delta) >= 0.0001:
                applied_adjustments[name] = delta
        for name, requested_delta in requested_adjustments.items():
            applied_delta = applied_adjustments.get(name, 0.0)
            clamp_delta = round(applied_delta - requested_delta, 4)
            if abs(clamp_delta) >= 0.0001:
                clamped_adjustments[name] = clamp_delta

    affect_snapshot = runtime_state.last_affect_head_snapshot
    if affect_snapshot:
        alignment_payload = {
            "event": "affect_alignment",
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "scores": affect_snapshot.get("scores"),
            "tags": affect_snapshot.get("tags"),
            "latency_ms": affect_snapshot.get("latency_ms"),
            "signals": {
                "valence": round(valence_signal, 4),
                "intimacy": round(intimacy_signal, 4),
                "tension": round(tension_signal, 4),
                "safety": round(float(input_safety), 4) if isinstance(input_safety, (int, float)) else None,
                "approach_avoid": round(float(input_approach), 4) if isinstance(input_approach, (int, float)) else None,
                "arousal": round(float(input_arousal), 4) if isinstance(input_arousal, (int, float)) else None,
            },
            "requested": requested_adjustments,
            "applied": applied_adjustments,
            "clamped": clamped_adjustments,
        }
        log_affect_head_event(AFFECT_ALIGNMENT_LOG, alignment_payload)

    return {
        "signals": {
            "valence_delta": round(valence, 4),
            "length_score": round(length_score, 4),
            "engagement_score": round(engagement, 4),
            "authenticity_score": round(authenticity, 4),
            "assistant_drift": round(assistant_drift, 4),
            "self_preoccupation": round(self_focus, 4),
            "affect_valence": round(affect_valence, 4),
            "affect_intimacy": round(affect_intimacy, 4),
            "affect_tension": round(affect_tension, 4),
            "affect_safety": round(float(input_safety), 4) if isinstance(input_safety, (int, float)) else None,
            "affect_arousal": round(float(input_arousal), 4) if isinstance(input_arousal, (int, float)) else None,
            "affect_approach_avoid": round(float(input_approach), 4) if isinstance(input_approach, (int, float)) else None,
            "affect_inhibition_social": round(float(input_inhib_social), 4) if isinstance(input_inhib_social, (int, float)) else None,
            "affect_inhibition_vulnerability": round(float(input_inhib_vuln), 4) if isinstance(input_inhib_vuln, (int, float)) else None,
            "affect_inhibition_self_restraint": round(float(input_inhib_self), 4) if isinstance(input_inhib_self, (int, float)) else None,
            "affect_expectedness": input_expectedness if isinstance(input_expectedness, str) else None,
            "affect_momentum_delta": input_momentum if isinstance(input_momentum, str) else None,
            "affect_intent": list(input_intents) if input_intents else None,
            "affect_rpe": round(float(input_rpe), 4) if isinstance(input_rpe, (int, float)) else None,
        },
        "requested": requested_adjustments,
        "applied": applied_adjustments,
        "clamped": clamped_adjustments,
        "pre": pre_hormones,
        "post": post_hormones,
    }


def _update_metric_history(
    reinforcement: dict[str, float],
    *,
    intent: str,
    length_label: str,
    profile: str,
    pre_hormones: Mapping[str, float] | None,
) -> dict[str, float]:
    authenticity = float(reinforcement.get("authenticity_score", 0.0))
    drift = float(reinforcement.get("assistant_drift", 0.0))
    self_focus = float(reinforcement.get("self_preoccupation", 0.0))
    affect_valence = float(reinforcement.get("affect_valence", 0.0))
    affect_intimacy = float(reinforcement.get("affect_intimacy", 0.0))
    affect_tension = float(reinforcement.get("affect_tension", 0.0))

    runtime_state.auth_history.append(authenticity)
    runtime_state.drift_history.append(drift)
    runtime_state.self_history.append(self_focus)
    runtime_state.affect_valence_history.append(affect_valence)
    runtime_state.affect_intimacy_history.append(affect_intimacy)
    runtime_state.affect_tension_history.append(affect_tension)
    runtime_state.metric_sample_counter += 1

    if (
        runtime_state.metric_sample_counter < 4
        or not runtime_state.auth_history
        or not runtime_state.drift_history
        or not runtime_state.self_history
    ):
        runtime_state.last_hormone_delta = {}
        return {}

    auth_avg = sum(runtime_state.auth_history) / len(runtime_state.auth_history)
    drift_avg = sum(runtime_state.drift_history) / len(runtime_state.drift_history)
    self_avg = sum(runtime_state.self_history) / len(runtime_state.self_history)
    valence_avg = (
        sum(runtime_state.affect_valence_history) / len(runtime_state.affect_valence_history)
        if runtime_state.affect_valence_history
        else 0.0
    )
    intimacy_avg = (
        sum(runtime_state.affect_intimacy_history) / len(runtime_state.affect_intimacy_history)
        if runtime_state.affect_intimacy_history
        else 0.0
    )
    tension_avg = (
        sum(runtime_state.affect_tension_history) / len(runtime_state.affect_tension_history)
        if runtime_state.affect_tension_history
        else 0.0
    )
    runtime_state.last_metric_averages = {
        "authenticity": round(auth_avg, 4),
        "assistant_drift": round(drift_avg, 4),
        "self_preoccupation": round(self_avg, 4),
        "affect_valence": round(valence_avg, 4),
        "affect_intimacy": round(intimacy_avg, 4),
        "affect_tension": round(tension_avg, 4),
    }

    if HORMONE_MODEL and pre_hormones:
        baseline_adjustments = HORMONE_MODEL.predict_delta(
            pre_hormones,
            reinforcement,
            intent=intent,
            length_label=length_label,
            profile=profile,
        )
    else:
        baseline_adjustments = _heuristic_hormone_adjustments(auth_avg, drift_avg, self_avg)

    if baseline_adjustments:
        state_engine.hormone_system.adjust_baseline(baseline_adjustments)
    runtime_state.metric_sample_counter = 0
    runtime_state.last_hormone_delta = dict(baseline_adjustments)
    return baseline_adjustments


def _reinforce_low_self_success(
    metrics: Mapping[str, float] | None,
    *,
    profile: str | None = None,
) -> None:
    """Carry momentum from low-self authenticity wins into the next turn."""
    if not metrics:
        runtime_state.low_self_success_streak = 0
        return
    auth = float(metrics.get("authenticity_score", 0.0) or 0.0)
    self_focus = float(metrics.get("self_preoccupation", 1.0) or 1.0)
    drift = float(metrics.get("assistant_drift", 1.0) or 1.0)
    outward = float(metrics.get("outward_streak_score", 0.0) or 0.0)
    profile_name = (profile or "").lower()
    drift_ceiling = LOW_SELF_SUCCESS_DRIFT_CEILING
    priming_multiplier = 1.0
    bonus_turns = 0
    outward_floor = OUTWARD_RELEASE_FLOOR
    if profile_name.startswith("instr"):
        drift_ceiling = LOW_SELF_INSTRUCT_DRIFT_CEILING
        priming_multiplier = LOW_SELF_INSTRUCT_PRIMING_MULTIPLIER
        bonus_turns = 1
        outward_floor = max(outward_floor, LOW_SELF_INSTRUCT_OUTWARD_FLOOR)
    elif profile_name.startswith("base"):
        priming_multiplier = LOW_SELF_BASE_PRIMING_MULTIPLIER
        bonus_turns = max(bonus_turns, LOW_SELF_BASE_EXTRA_TURNS)
        outward_floor = max(outward_floor, LOW_SELF_BASE_OUTWARD_FLOOR)
    if (
        auth >= LOW_SELF_SUCCESS_AUTH
        and self_focus <= LOW_SELF_SUCCESS_MAX_SELF
        and drift <= drift_ceiling
    ):
        runtime_state.low_self_success_streak += 1
        effective_outward = max(outward, outward_floor)
        priming_base = LOW_SELF_SUCCESS_PRIMING + max(0.0, effective_outward - OUTWARD_RELEASE_FLOOR) * 0.4
        priming_boost = priming_base * priming_multiplier
        if runtime_state.low_self_success_streak == 1:
            runtime_state.reset_priming_bias = max(runtime_state.reset_priming_bias, round(priming_boost * 0.65, 3))
            runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 1 + bonus_turns)
        elif runtime_state.low_self_success_streak >= LOW_SELF_SUCCESS_STREAK_TARGET:
            runtime_state.reset_priming_bias = max(runtime_state.reset_priming_bias, round(priming_boost, 3))
            runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 2 + bonus_turns)
    else:
        runtime_state.low_self_success_streak = 0



def _count_sentences(text: str) -> int:
    if not text or not text.strip():
        return 0
    parts = [segment for segment in re.split(r"[.!?]+", text) if segment.strip()]
    return len(parts)

def _select_intent(user_message: str, *, context: dict[str, Any]) -> IntentPrediction:
    memory_summary = ""
    memory_block = context.get("memory")
    if isinstance(memory_block, dict):
        memory_summary = str(memory_block.get("summary") or "")
    prediction = predict_intent(user_message, context_summary=memory_summary)
    return prediction


def _init_local_llama() -> LocalLlamaEngine | None:
    if os.getenv("LLAMA_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        logger.info("Local llama engine disabled via LLAMA_DISABLE.")
        return None
    if not LLAMA_SERVER_BIN or not LLAMA_MODEL_PATH:
        logger.info(
            "Local llama engine disabled (missing LLAMA_SERVER_BIN or LLAMA_MODEL_PATH); "
            "falling back to remote or heuristic responses."
        )
        return None
    try:
        return LocalLlamaEngine(
            Path(LLAMA_SERVER_BIN),
            Path(LLAMA_MODEL_PATH),
            host=LLAMA_SERVER_HOST,
            port=LLAMA_SERVER_PORT,
            model_alias=LLAMA_MODEL_ALIAS,
            extra_args=LLAMA_SERVER_ARGS,
            timeout=LLAMA_SERVER_TIMEOUT,
            readiness_timeout=LLAMA_READINESS_TIMEOUT,
            max_tokens=LLAMA_COMPLETION_TOKENS,
        )
    except Exception as exc:  # pragma: no cover - configuration errors
        logger.warning("Failed to initialize local llama engine: %s", exc)
        return None


async def _shutdown_clients() -> None:
    """Close active LLM clients and stop the local engine."""
    global llm_client, local_llama_engine, AFFECT_SIDECAR
    if llm_client is not None:
        await llm_client.aclose()
        llm_client = None
    if local_llama_engine is not None:
        await local_llama_engine.stop()
        local_llama_engine = None
    if AFFECT_SIDECAR is not None:
        try:
            await AFFECT_SIDECAR.stop()
        except Exception:
            pass


async def _configure_clients() -> None:
    """Instantiate LLM clients and ensure the local engine is running."""
    global llm_client, local_llama_engine
    await _shutdown_clients()
    if LLM_ENDPOINT:
        llm_client = LivingLLMClient(LLM_ENDPOINT, timeout=LLM_TIMEOUT)
        logger.info("Configured living_llm endpoint at %s", LLM_ENDPOINT)
    else:
        logger.info("No living_llm endpoint configured; heuristic replies enabled.")
    candidate_engine = _init_local_llama()
    if candidate_engine is not None:
        local_llama_engine = candidate_engine
        try:
            await local_llama_engine.ensure_started()
            logger.info(
                "Local llama engine ready at %s (alias=%s)",
                local_llama_engine.base_url,
                local_llama_engine.model_alias,
            )
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("Local llama engine failed to start: %s", exc)
            await local_llama_engine.stop()
            local_llama_engine = None


async def _reload_runtime_settings() -> dict[str, Any]:
    """Reload configuration, restart clients, and report active caps."""
    clear_settings_cache()
    _refresh_settings()
    await _configure_clients()
    return {
        "llm_endpoint": LLM_ENDPOINT,
        "llm_timeout": LLM_TIMEOUT,
        "local_engine": local_llama_engine is not None,
        "llama": {
            "model_path": LLAMA_MODEL_PATH,
            "host": LLAMA_SERVER_HOST,
            "port": LLAMA_SERVER_PORT,
            "completion_tokens": LLAMA_COMPLETION_TOKENS,
        },
        "base_sampling": {
            "temperature": BASE_TEMPERATURE,
            "top_p": BASE_TOP_P,
            "frequency_penalty": BASE_FREQUENCY_PENALTY,
        },
        "length_overrides": {
            label: overrides.get("max_tokens")
            for label, overrides in LENGTH_SAMPLING_OVERRIDES.items()
        },
    }

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# -------------------- Affect head console helper -------------------- #

def _ensure_affect_head_console() -> None:
    """Open a compact console window tailing the affect head telemetry."""
    global affect_head_console
    if os.name != "nt":
        return
    if affect_head_console and affect_head_console.poll() is None:
        return
    if os.getenv("AFFECT_HEAD_CONSOLE", "1").strip().lower() in {"0", "false", "off", "disable", "disabled"}:
        return
    log_path = BASE_DIR / "logs" / "affect_head_raw.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        try:
            log_path.touch()
        except Exception:
            pass
    python_exe = str(Path(sys.executable or "python"))
    monitor_path = BASE_DIR / "scripts" / "affect_head_monitor.py"
    cmd: list[str] = [python_exe, str(monitor_path), "--log", str(log_path)]
    proc_kwargs: dict[str, Any] = {"cwd": str(BASE_DIR)}
    if os.name == "nt":
        # Match other telemetry consoles (conhost-based) instead of forcing PowerShell.
        proc_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        proc_kwargs["start_new_session"] = True
    try:
        affect_head_console = subprocess.Popen(cmd, **proc_kwargs)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("Failed to open affect head console: %s", exc)


class EventPayload(BaseModel):
    """Schema describing an external stimulus event."""

    content: str = Field(..., min_length=1, description="Narrative summary of the event.")
    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Relative importance used during memory consolidation.",
    )
    stimulus: str | None = Field(
        default=None,
        description="Optional stimulus keyword affecting hormone levels.",
    )


class ChatMessage(BaseModel):
    """Schema describing messages exchanged via the chat UI."""

    message: str = Field(..., min_length=1, max_length=2000)
    stimulus: str | None = Field(
        default=None,
        description="Optional stimulus keyword applied alongside the user message.",
    )


class ModelSwitchRequest(BaseModel):
    """Schema for switching between model configuration profiles."""

    profile: str = Field(..., min_length=1, description="Model profile identifier (e.g. 'instruct', 'base').")


class SessionResetRequest(BaseModel):
    """Schema for resetting the in-memory conversational session."""

    reason: str | None = Field(default=None, description="Optional operator note explaining why the session was reset.")
    keep_metric_history: bool = Field(
        default=False,
        description="If true, retain aggregate probe metrics instead of clearing rolling histories.",
    )


async def run_state_updates() -> None:
    """Continuously advance the state engine on a fixed interval."""
    while True:
        await state_engine.tick()
        await asyncio.sleep(state_engine.tick_interval)


def _auto_telemetry_enabled() -> bool:
    flag = os.getenv("LIVING_AUTO_TELEMETRY", "1").strip().lower()
    return flag not in {"0", "false", "off", "disable", "disabled"}


def _telemetry_console_visible() -> bool:
    flag = os.getenv("LIVING_TELEMETRY_CONSOLE", "0").strip().lower()
    return flag not in {"0", "false", "off", "disable", "disabled"}


async def _ensure_telemetry_monitor() -> None:
    """Launch the live telemetry console if not already running."""
    global telemetry_process
    if not _auto_telemetry_enabled():
        return
    if telemetry_process and telemetry_process.returncode is None:
        return
    python_exe = sys.executable or "python"
    cmd = [python_exe, "-m", "scripts.live_telemetry"]
    proc_kwargs: dict[str, Any] = {"cwd": str(BASE_DIR)}
    if os.name == "nt":
        if _telemetry_console_visible():
            proc_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        else:
            proc_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        proc_kwargs["start_new_session"] = True
    try:
        await asyncio.sleep(0.5)
        telemetry_process = await asyncio.create_subprocess_exec(*cmd, **proc_kwargs)
        logger.info("Launched live telemetry console (pid=%s)", telemetry_process.pid)
    except Exception as exc:  # pragma: no cover - best effort
        telemetry_process = None
        logger.warning("Failed to launch live telemetry console: %s", exc)


async def _stop_telemetry_monitor() -> None:
    """Terminate the live telemetry console if it was spawned."""
    global telemetry_process
    if not telemetry_process:
        return
    if telemetry_process.returncode is None:
        telemetry_process.terminate()
        try:
            await asyncio.wait_for(telemetry_process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            telemetry_process.kill()
        except Exception:
            pass
    logger.info("Live telemetry console closed")
    telemetry_process = None


@app.on_event("startup")
async def start_background_tasks() -> None:
    """Initialize the background state update loop when the app starts."""
    global update_task
    if update_task is None or update_task.done():
        update_task = asyncio.create_task(run_state_updates())
    await _configure_clients()
    # Warm the affect sidecar if configured
    if AFFECT_SIDECAR is not None:
        try:
            await AFFECT_SIDECAR.ensure_running()
            logger.info("Affect sidecar ready at %s", AFFECT_SIDECAR.base_url)
            ready_payload = {
                "event": "affect_head_ready",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "base_url": AFFECT_SIDECAR.base_url,
                "model_path": str(getattr(AFFECT_SIDECAR, "config", {}).get("model_path", "")),
                "source": "affect_head",
            }
            log_affect_head_event(AFFECT_HEAD_TELEMETRY_LOG, ready_payload)
            append_affect_raw(BASE_DIR / "logs", ready_payload)
            append_affect_compact(BASE_DIR / "logs", ready_payload)
            runtime_state.last_affect_head_snapshot = ready_payload
        except Exception as exc:
            logger.warning("Affect sidecar failed to start: %s", exc)
    _ensure_affect_head_console()
    await _ensure_telemetry_monitor()


@app.on_event("shutdown")
async def stop_background_tasks() -> None:
    """Cancel the background update loop when the app stops."""
    global update_task, affect_head_console
    if update_task is not None:
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        update_task = None
    await _shutdown_clients()
    await _stop_telemetry_monitor()
    if affect_head_console and affect_head_console.poll() is None:
        affect_head_console.terminate()
    if affect_head_console:
        try:
            affect_head_console.wait(timeout=1.0)
        except Exception:
            try:
                affect_head_console.kill()
            except Exception:
                pass
        affect_head_console = None


@app.get("/admin/model")
async def get_active_model() -> dict[str, Any]:
    """Return the currently active model profile and settings file."""
    return {
        "profile": current_profile(),
        "settings_file": current_settings_file(),
    }


@app.post("/admin/model")
async def switch_model(request: ModelSwitchRequest) -> dict[str, Any]:
    """Switch between model configuration profiles and reload runtime settings."""
    try:
        settings_file = resolve_settings_file(request.profile)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=str(exc))
    os.environ["LIVING_SETTINGS_FILE"] = settings_file
    config = await _reload_runtime_settings()
    return {
        "profile": current_profile(),
        "settings_file": settings_file,
        "settings": config,
    }


@app.post("/session/reset")
async def reset_session(request: SessionResetRequest) -> dict[str, Any]:
    """Reset the active conversational session without stopping the server."""
    return await _reset_session_state(
        reason=request.reason,
        keep_metric_history=request.keep_metric_history,
    )


@app.get("/telemetry/snapshot")
async def telemetry_snapshot() -> dict[str, Any]:
    """Expose a one-off telemetry snapshot for dashboards and CLI monitors."""
    return compose_live_status(
        state_engine=state_engine,
        runtime_state=runtime_state,
        model_alias=LLAMA_MODEL_ALIAS,
        local_llama_available=local_llama_engine is not None,
        affect_head_snapshot=runtime_state.last_affect_head_snapshot,
    )


@app.get("/telemetry/stream")
async def telemetry_stream() -> StreamingResponse:
    """Yield Server-Sent Events with live telemetry every configured interval."""
    interval = max(0.2, TELEMETRY_INTERVAL_SECONDS)

    async def event_generator() -> AsyncIterator[str]:
        try:
            while True:
                payload = compose_live_status(
                    state_engine=state_engine,
                    runtime_state=runtime_state,
                    model_alias=LLAMA_MODEL_ALIAS,
                    local_llama_available=local_llama_engine is not None,
                    affect_head_snapshot=runtime_state.last_affect_head_snapshot,
                )
                yield _sse_event("telemetry", payload)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _build_affect_context() -> dict[str, Any]:
    """Return the affect overview if the feature is enabled."""
    if not AFFECT_CONTEXT_ENABLED:
        return {}
    return state_engine.affect_overview()


def _build_chat_context(*, long_term_limit: int = 5) -> dict[str, Any]:
    """Compatibility wrapper that proxies to the shared chat-context builder."""
    persona_snapshot = build_persona_snapshot(state_engine)
    affect_context = _build_affect_context()
    return build_chat_context(
        state_engine=state_engine,
        local_llama_engine=local_llama_engine,
        persona_snapshot=persona_snapshot,
        affect_context=affect_context,
        affect_memory_preview_enabled=AFFECT_MEMORY_PREVIEW_ENABLED,
        recency_window_seconds=AFFECT_RECENCY_WINDOW_SECONDS,
        self_narration_note=runtime_state.self_narration_note,
        long_term_limit=long_term_limit,
    )



def _compose_heuristic_reply(
    user_message: str,
    *,
    context: Mapping[str, Any],
    intent: str,
    length_plan: Mapping[str, Any],
) -> str:
    """Compatibility shim for tests that relied on the legacy helper."""
    return compose_heuristic_reply(
        user_message,
        context=context,
        intent=intent,
        length_plan=length_plan,
        state_engine=state_engine,
        shorten=_shorten,
    )



def _classify_user_affect(message: str) -> AffectClassification | None:
    """Run the affect classifier for the incoming message."""
    text = (message or "").strip()
    if not text:
        return None
    if AFFECT_CLASSIFIER is None:
        _reinitialize_affect_classifier()
    classifier = AFFECT_CLASSIFIER
    if classifier is None:
        return None
    try:
        # Use current user text for CAHM scoring to avoid prompt dilution.
        result = classifier.classify(text)
        runtime_state.recent_turns.append(("user", text))
        # Log structured affect-head telemetry if available
        if result:
            engine = (result.metadata or {}).get("source") or "heuristic"
            extras: dict[str, Any] = {}
            for key in [
                "safety",
                "arousal",
                "approach_avoid",
                "inhibition_social",
                "inhibition_vulnerability",
                "inhibition_self_restraint",
                "expectedness",
                "momentum_delta",
                "affection_subtype",
                "rpe",
                "rationale",
            ]:
                value = getattr(result, key, None)
                if value is not None:
                    extras[key] = value
            if getattr(result, "intent", None):
                extras["intent"] = list(result.intent)
            payload = {
                "event": "affect_classification",
                "text_preview": _shorten(text, 120),
                "source": "affect_head",
                "engine": engine,
                "scores": {
                    "valence": result.valence,
                    "intimacy": result.intimacy,
                    "tension": result.tension,
                    "confidence": result.confidence,
                },
                "tags": list(result.tags),
                "latency_ms": (result.metadata or {}).get("latency_ms"),
                # hide raw_completion to avoid leaking prompt snippets into memory/telemetry
                "raw_completion": None,
                "reasoning": (result.metadata or {}).get("reasoning"),
                "rationale": (result.metadata or {}).get("rationale"),
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            }
            if extras:
                payload["extras"] = extras
            log_affect_head_event(AFFECT_HEAD_TELEMETRY_LOG, payload)
            append_affect_raw(BASE_DIR / "logs", payload)
            append_affect_compact(BASE_DIR / "logs", payload)
            runtime_state.last_affect_head_snapshot = payload
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Affect classifier failed: %s", exc)
        return None
    log_affect_classification(
        text,
        result,
        AFFECT_CLASSIFIER_LOG,
        shorten=_shorten,
        logger=logger,
    )
    return result


def _merge_affect_signal(
    existing: float | None,
    classified: float,
    confidence: float,
) -> float | None:
    """Blend classifier output into an existing signal when confidence is high enough."""
    blend_weight = max(0.0, min(1.0, confidence))
    if blend_weight < AFFECT_CLASSIFIER_BLEND_MIN_CONFIDENCE:
        return existing
    if existing is None:
        return round(classified, 4)
    return round(((1.0 - blend_weight) * float(existing)) + (blend_weight * classified), 4)



def _prepare_chat_request(
    user_message: str,
    *,
    user_affect: AffectClassification | None = None,
    helper_penalty_scale: float | None = None,
    helper_penalty_reason: str | None = None,
) -> tuple[dict[str, Any], IntentPrediction, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build context, predictions, and sampling parameters for a chat turn."""
    persona_snapshot = build_persona_snapshot(state_engine)
    if runtime_state.self_narration_note:
        persona_snapshot["internal_note"] = runtime_state.self_narration_note
    affect_context_snapshot = _build_affect_context()
    context = build_chat_context(
        state_engine=state_engine,
        local_llama_engine=local_llama_engine,
        persona_snapshot=persona_snapshot,
        affect_context=affect_context_snapshot,
        affect_memory_preview_enabled=AFFECT_MEMORY_PREVIEW_ENABLED,
        recency_window_seconds=AFFECT_RECENCY_WINDOW_SECONDS,
        self_narration_note=runtime_state.self_narration_note,
    )
    affect_dict: dict[str, Any] | None = None
    if user_affect:
        affect_dict = user_affect.as_dict()
        context["user_affect"] = affect_dict
    intent_prediction = _select_intent(user_message, context=context)
    length_plan = plan_response_length(user_message, intent_prediction.intent)
    context["intent"] = intent_prediction.intent
    context["intent_confidence"] = intent_prediction.confidence
    context["intent_rationale"] = intent_prediction.rationale
    context["length_plan"] = length_plan

    policy_snapshot: SamplingPolicy | None = None
    sampling: dict[str, Any] = {}
    traits = state_engine.trait_snapshot()
    if AFFECT_SAMPLING_PREVIEW_ENABLED and traits:
        policy_snapshot = derive_policy(traits)
        policy_kwargs = policy_snapshot.as_kwargs()
        context["sampling_policy_preview"] = dict(policy_kwargs)
        sampling.update(policy_kwargs)

    hormones = context.get("hormones", {}) or {}
    if hormones:
        hormone_sampling, hormone_style_hits = sampling_params_from_hormones(
            hormones,
            base_temperature=BASE_TEMPERATURE,
            base_top_p=BASE_TOP_P,
            base_frequency_penalty=BASE_FREQUENCY_PENALTY,
            hormone_style_map=HORMONE_STYLE_MAP,
            max_completion_tokens=LLAMA_COMPLETION_TOKENS,
        )
    else:
        hormone_sampling, hormone_style_hits = {}, []
    if sampling and hormone_sampling:
        for key, value in hormone_sampling.items():
            if key in {"temperature", "top_p", "frequency_penalty", "presence_penalty"}:
                base_value = float(sampling.get(key, value))
                blended = (
                    (1.0 - AFFECT_SAMPLING_BLEND_WEIGHT) * base_value
                    + AFFECT_SAMPLING_BLEND_WEIGHT * float(value)
                )
                sampling[key] = round(blended, 4)
            else:
                sampling[key] = value
    elif hormone_sampling:
        sampling.update(hormone_sampling)

    if traits:
        sampling = inject_self_observation_bias(
            runtime_state,
            sampling,
            traits,
            base_temperature=BASE_TEMPERATURE,
            base_top_p=BASE_TOP_P,
            base_frequency_penalty=BASE_FREQUENCY_PENALTY,
        )

    sampling = apply_intent_sampling(
        sampling,
        intent_prediction.intent,
        base_temperature=BASE_TEMPERATURE,
        base_top_p=BASE_TOP_P,
        base_frequency_penalty=BASE_FREQUENCY_PENALTY,
    )
    sampling = apply_length_sampling(
        sampling,
        length_plan,
        base_temperature=BASE_TEMPERATURE,
        base_top_p=BASE_TOP_P,
        base_frequency_penalty=BASE_FREQUENCY_PENALTY,
    )
    sampling, affect_overrides = apply_affect_style_overrides(
        sampling,
        user_affect,
        base_temperature=BASE_TEMPERATURE,
        base_top_p=BASE_TOP_P,
        base_frequency_penalty=BASE_FREQUENCY_PENALTY,
        max_completion_tokens=LLAMA_COMPLETION_TOKENS,
    )
    affect_min_tokens = None
    if affect_overrides and isinstance(affect_overrides.get("min_tokens_floor"), (int, float)):
        affect_min_tokens = int(affect_overrides["min_tokens_floor"])
    length_plan_floor = None
    plan_max_tokens = length_plan.get("max_tokens")
    if isinstance(plan_max_tokens, (int, float)):
        length_plan_floor = int(plan_max_tokens)
    combined_min_tokens = None
    candidates = [value for value in (affect_min_tokens, length_plan_floor) if isinstance(value, int)]
    if candidates:
        combined_min_tokens = max(candidates)
    profile = current_profile()
    active_tags = gather_active_tags(state_engine)
    controller_features = build_controller_feature_map(
        state_engine=state_engine,
        runtime_state=runtime_state,
        traits=traits,
        hormones=hormones,
        intent=intent_prediction.intent,
        length_label=length_plan.get("label"),
        profile=profile,
        tags=active_tags,
    )
    controller_step = run_controller_policy(
        CONTROLLER_RUNTIME,
        CONTROLLER_LOCK,
        runtime_state,
        controller_features,
        active_tags,
    )
    controller_snapshot: dict[str, Any] | None = None
    controller_applied: dict[str, Any] = {}
    if controller_step:
        sampling, controller_applied = apply_controller_adjustments(
            runtime_state,
            sampling,
            controller_step.adjustments,
            base_temperature=BASE_TEMPERATURE,
            base_top_p=BASE_TOP_P,
            base_frequency_penalty=BASE_FREQUENCY_PENALTY,
            max_completion_tokens=LLAMA_COMPLETION_TOKENS,
            min_tokens_floor=combined_min_tokens,
            reset_session=_reset_live_session,
        )
        applied_serialized: dict[str, Any] = {}
        for key, value in controller_applied.items():
            if isinstance(value, (int, float)):
                applied_serialized[key] = value if isinstance(value, int) else round(float(value), 6)
            else:
                applied_serialized[key] = value
        controller_snapshot = {
            "adjustments": {key: round(float(value), 6) for key, value in controller_step.adjustments.items()},
            "hidden_state": [round(float(value), 6) for value in controller_step.hidden_state],
            "raw_outputs": [round(float(value), 6) for value in controller_step.raw_outputs],
            "applied": applied_serialized,
        }
    if combined_min_tokens:
        current_tokens = int(sampling.get("max_tokens", LLAMA_COMPLETION_TOKENS) or LLAMA_COMPLETION_TOKENS)
        if current_tokens < combined_min_tokens:
            sampling["max_tokens"] = combined_min_tokens
            controller_applied["max_tokens"] = combined_min_tokens
    helper_penalty_meta: dict[str, Any] | None = None
    helper_penalty_value = float(helper_penalty_scale or 0.0)
    if helper_penalty_value > 1e-4:
        helper_penalty_meta = {
            "scale": round(helper_penalty_value, 3),
        }
        if helper_penalty_reason:
            helper_penalty_meta["reason"] = helper_penalty_reason
        sampling = apply_helper_tone_bias(sampling, helper_penalty_value)
        controller_applied.setdefault("helper_penalty_override", helper_penalty_meta["scale"])

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sampling": dict(sampling),
        "policy_preview": policy_snapshot.as_kwargs() if policy_snapshot else None,
        "hormone_sampling": hormone_sampling,
        "intent": intent_prediction.intent,
        "length_label": length_plan.get("label"),
        "profile": profile,
    }
    if runtime_state.last_affect_head_snapshot:
        snapshot["affect_head"] = dict(runtime_state.last_affect_head_snapshot)
    if controller_snapshot:
        snapshot["controller"] = controller_snapshot
        snapshot["controller_input"] = {
            "features": {key: round(float(value), 6) for key, value in controller_features.items()},
            "tags": list(active_tags),
        }
    elif controller_features:
        snapshot["controller_input"] = {
            "features": {key: round(float(value), 6) for key, value in controller_features.items()},
            "tags": list(active_tags),
        }
    if helper_penalty_meta:
        snapshot["helper_penalty"] = helper_penalty_meta
    if hormone_style_hits:
        snapshot["hormone_style_hits"] = hormone_style_hits
    if affect_overrides:
        snapshot["affect_overrides"] = affect_overrides
    if affect_dict:
        snapshot["user_affect"] = affect_dict
    if AFFECT_DEBUG_PANEL_ENABLED:
        runtime_state.last_sampling_snapshot = snapshot
    if AFFECT_SAMPLING_PREVIEW_ENABLED or AFFECT_DEBUG_PANEL_ENABLED:
        log_sampling_snapshot(snapshot, SAMPLING_SNAPSHOT_LOG, logger=logger)
    memory_spotlight = []
    memory_block = context.get("memory") or {}
    spotlight_entries = memory_block.get("spotlight") if isinstance(memory_block, dict) else None
    if isinstance(spotlight_entries, list):
        for entry in spotlight_entries:
            if isinstance(entry, Mapping):
                key = entry.get("key")
                if key:
                    memory_spotlight.append(str(key))
    runtime_state.memory_spotlight_keys = memory_spotlight
    llm_context = {
        key: value
        for key, value in context.items()
        if key not in {"hormones", "persona", "affect", "sampling_policy_preview"}
    }
    llm_context["intent_prompt"] = intent_prompt_fragment(intent_prediction.intent)
    llm_context["length_prompt"] = length_plan.get("prompt")
    llm_context["intent"] = intent_prediction.intent
    llm_context["intent_confidence"] = intent_prediction.confidence
    llm_context["length_plan"] = length_plan
    if affect_dict:
        llm_context["user_affect"] = affect_dict
    return context, intent_prediction, length_plan, sampling, llm_context, snapshot


def _sse_event(event: str, data: Any) -> str:
    """Serialize an SSE event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _reset_session_state(
    reason: str | None = None,
    *,
    keep_metric_history: bool = False,
) -> dict[str, Any]:
    """Reset in-memory session state, hormones, and rolling metrics."""
    state_engine.reset()
    runtime_state.session_counter += 1
    if not keep_metric_history:
        runtime_state.clear_metric_state()
    runtime_state.reset_controller()
    runtime_state.last_hormone_delta = None
    runtime_state.self_focus_streak = 0
    runtime_state.clamp_recovery_turns = 0
    runtime_state.clamp_priming_turns = 0
    runtime_state.recovery_good_streak = 0
    runtime_state.reset_priming_bias = 0.0
    runtime_state.recovery_lowself_streak = 0
    runtime_state.last_user_prompt = ""
    runtime_state.low_self_success_streak = 0
    runtime_state.helper_drift_level = 0.0
    runtime_state.memory_spotlight_keys = []
    reset_outward_streak(runtime_state.reinforcement_tracker)
    status = compose_live_status(
        state_engine=state_engine,
        runtime_state=runtime_state,
        model_alias=LLAMA_MODEL_ALIAS,
        local_llama_available=local_llama_engine is not None,
    )
    status["reset_reason"] = reason
    logger.info(
        "Session reset -> id=%s reason=%s keep_metrics=%s",
        status.get("session_id"),
        reason or "unspecified",
        keep_metric_history,
    )
    return status


async def _reset_live_session(reason: str) -> None:
    """Background wrapper so controller clamps can trigger a clean reset."""
    try:
        await _reset_session_state(reason=reason, keep_metric_history=False)
        if reason == "controller_clamp":
            focus_phrase = extract_focus_phrase(runtime_state.last_user_prompt or "")
            if focus_phrase:
                safe_focus = focus_phrase.replace('"', "")
                priming_text = (
                    f'priming: literally begin the next reply with "You mentioned {safe_focus}" '
                    "and stay with their experience for two sentences before naming your own body."
                )
            else:
                priming_text = (
                    "priming: quote their exact words, start with \"You...\" or \"We...\", "
                    "and keep the first two sentences anchored to them before any I-statements."
                )
            await state_engine.register_event(
                content=priming_text,
                strength=0.85,
                stimulus_type="affection",
                apply_sentiment=False,
            )
            await state_engine.register_event(
                content="priming: first sentence must start with you/we and echo their phrasing before reflecting inward.",
                strength=0.75,
                stimulus_type="reward",
                apply_sentiment=False,
            )
            runtime_state.clamp_priming_turns = max(runtime_state.clamp_priming_turns, 3)
            runtime_state.reset_priming_bias = RESET_PRIMING_BIAS_DEFAULT
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Live session reset failed for reason=%s", reason)


async def _generate_chat_reply(
    user_message: str,
    *,
    user_affect: AffectClassification | None = None,
    regen_attempt: int = 0,
    helper_bias_override: float | None = None,
    regen_trace: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any] | None, IntentPrediction, dict[str, Any], dict[str, Any]]:
    """Generate a reply via the LLM bridge, falling back to heuristics if needed."""
    penalty_reason = "voice_guard_regen" if helper_bias_override else None
    (
        context,
        intent_prediction,
        length_plan,
        sampling,
        llm_context,
        snapshot,
    ) = _prepare_chat_request(
        user_message,
        user_affect=user_affect,
        helper_penalty_scale=helper_bias_override,
        helper_penalty_reason=penalty_reason,
    )
    telemetry = compose_turn_telemetry(
        context=context,
        sampling=sampling,
        snapshot=snapshot,
        state_engine=state_engine,
        shorten=_shorten,
        model_alias=LLAMA_MODEL_ALIAS,
    )
    if user_affect:
        telemetry["user_affect"] = user_affect.as_dict()

    if local_llama_engine is not None:
        try:
            reply_text, payload = await local_llama_engine.generate_reply(
                user_message, llm_context, sampling=sampling
            )
            if reply_text.strip():
                telemetry["engine"] = "local"
                telemetry["llm_metrics"] = payload.get("timings") if isinstance(payload, dict) else None
                return reply_text, "local", payload, intent_prediction, length_plan, telemetry
            logger.warning("Local llama engine returned empty reply; using fallback.")
        except Exception as exc:  # pragma: no cover - runtime or bridge errors
            logger.warning("Local llama engine failed, using fallback: %s", exc)
    if llm_client is None or not LLM_ENDPOINT:
        telemetry["engine"] = "heuristic"
        reply_text = compose_heuristic_reply(
            user_message,
            context=context,
            intent=intent_prediction.intent,
            length_plan=length_plan,
            state_engine=state_engine,
            shorten=_shorten,
        )
        source = "heuristic"
        payload = None
    else:
        try:
            reply_text, payload = await llm_client.generate_reply(user_message, llm_context)
            if reply_text.strip():
                telemetry["engine"] = "remote_llm"
                source = "llm"
            else:
                logger.warning("living_llm returned an empty reply; using heuristic fallback.")
                reply_text = ""
                source = "llm"
        except Exception as exc:  # pragma: no cover - network or endpoint errors
            logger.warning("living_llm request failed, using fallback: %s", exc)
            reply_text = ""
            payload = None
            source = "llm"
        if not reply_text:
            telemetry["engine"] = "heuristic"
            reply_text = compose_heuristic_reply(
                user_message,
                context=context,
                intent=intent_prediction.intent,
                length_plan=length_plan,
                state_engine=state_engine,
                shorten=_shorten,
            )
            source = "heuristic"
            payload = None
    guard_verdict = VOICE_GUARD.evaluate(reply_text)
    if (
        guard_verdict.flagged
        and regen_attempt < MAX_HELPER_REGEN_ATTEMPTS
        and reply_text.strip()
        and source != "heuristic"
    ):
        regen_meta = dict(regen_trace or {})
        regen_meta["attempts"] = regen_attempt + 1
        regen_meta["last_score"] = round(guard_verdict.score, 3)
        regen_meta["source"] = source
        helper_boost = max(helper_bias_override or 0.0, 0.85 + guard_verdict.score * 0.5)
        return await _generate_chat_reply(
            user_message,
            user_affect=user_affect,
            regen_attempt=regen_attempt + 1,
            helper_bias_override=helper_boost,
            regen_trace=regen_meta,
        )
    if regen_trace:
        telemetry["voice_guard_regen"] = dict(regen_trace)
    return reply_text, source, payload, intent_prediction, length_plan, telemetry


async def _finalize_chat_response(
    user_message: str,
    reply_text: str,
    source: str,
    llm_payload: dict[str, Any] | None,
    intent_prediction: IntentPrediction,
    length_plan: dict[str, Any],
    telemetry: dict[str, Any] | None,
    *,
    user_affect: AffectClassification | None = None,
) -> dict[str, Any]:
    """Apply reinforcement, update state, log turn diagnostics, and format the API response."""
    runtime_state.last_user_prompt = user_message or ""
    runtime_state.recent_turns.append(("assistant", reply_text or ""))
    active_profile = telemetry.get("profile") if telemetry else current_profile()
    reply_echo = _shorten(reply_text, 200)
    short_user = _shorten(user_message, 200)
    ai_content = f"I replied in my own voice {reply_echo}"
    await state_engine.register_event(
        ai_content,
        strength=0.4,
        mood=state_engine.state["mood"],
        apply_sentiment=False,
    )
    reinforcement = score_response(
        user_message,
        reply_text,
        tracker=runtime_state.reinforcement_tracker,
    )
    if user_affect:
        reinforcement["input_affect_valence"] = round(user_affect.valence, 4)
        reinforcement["input_affect_intimacy"] = round(user_affect.intimacy, 4)
        reinforcement["input_affect_tension"] = round(user_affect.tension, 4)
        reinforcement["affect_classifier_confidence"] = round(user_affect.confidence, 4)
        if user_affect.tags:
            reinforcement["affect_classifier_tags"] = list(user_affect.tags)
        # propagate extended CAHM fields for downstream mapping
        if user_affect.safety is not None:
            reinforcement["input_affect_safety"] = round(user_affect.safety, 4)
        if user_affect.arousal is not None:
            reinforcement["input_affect_arousal"] = round(user_affect.arousal, 4)
        if user_affect.approach_avoid is not None:
            reinforcement["input_affect_approach_avoid"] = round(user_affect.approach_avoid, 4)
        if user_affect.inhibition_social is not None:
            reinforcement["input_affect_inhibition_social"] = round(user_affect.inhibition_social, 4)
        if user_affect.inhibition_vulnerability is not None:
            reinforcement["input_affect_inhibition_vulnerability"] = round(user_affect.inhibition_vulnerability, 4)
        if user_affect.inhibition_self_restraint is not None:
            reinforcement["input_affect_inhibition_self_restraint"] = round(user_affect.inhibition_self_restraint, 4)
        if user_affect.expectedness:
            reinforcement["input_affect_expectedness"] = user_affect.expectedness
        if user_affect.momentum_delta:
            reinforcement["input_affect_momentum_delta"] = user_affect.momentum_delta
        if user_affect.intent:
            reinforcement["input_affect_intent"] = list(user_affect.intent)
        if user_affect.affection_subtype:
            reinforcement["input_affect_affection_subtype"] = user_affect.affection_subtype
        if user_affect.rpe is not None:
            reinforcement["input_affect_rpe"] = round(user_affect.rpe, 4)
        if user_affect.rationale:
            reinforcement["affect_classifier_rationale"] = user_affect.rationale
        valence_blend = _merge_affect_signal(
            reinforcement.get("affect_valence"),
            user_affect.valence,
            user_affect.confidence,
        )
        intimacy_blend = _merge_affect_signal(
            reinforcement.get("affect_intimacy"),
            user_affect.intimacy,
            user_affect.confidence,
        )
        tension_blend = _merge_affect_signal(
            reinforcement.get("affect_tension"),
            user_affect.tension,
            user_affect.confidence,
        )
        if valence_blend is not None:
            reinforcement["affect_valence"] = valence_blend
        if intimacy_blend is not None:
            reinforcement["affect_intimacy"] = intimacy_blend
        if tension_blend is not None:
            reinforcement["affect_tension"] = tension_blend
    voice_guard_verdict = VOICE_GUARD.evaluate(reply_text)
    voice_guard_dict = voice_guard_verdict.to_dict()
    reinforcement["voice_guard_score"] = round(voice_guard_verdict.score, 4)
    reinforcement["voice_guard_flagged"] = voice_guard_verdict.flagged
    if voice_guard_verdict.flagged:
        penalty = min(1.0, 0.6 + voice_guard_verdict.score * 0.4)
        reinforcement["assistant_drift"] = max(
            float(reinforcement.get("assistant_drift", 0.0) or 0.0),
            penalty,
        )
        auth_score = reinforcement.get("authenticity_score")
        if isinstance(auth_score, (int, float)):
            reinforcement["authenticity_score"] = round(max(0.0, auth_score - 0.2), 4)
        runtime_state.helper_drift_level = min(1.0, runtime_state.helper_drift_level + HELPER_PENALTY_STEP)
    else:
        runtime_state.helper_drift_level = max(0.0, runtime_state.helper_drift_level - HELPER_PENALTY_DECAY)
    if telemetry is not None:
        telemetry["voice_guard"] = voice_guard_dict
    pre_snapshot = telemetry.get("pre", {}) if telemetry else {}
    hormone_trace = _apply_reinforcement_signals(
        reinforcement,
        length_plan=length_plan,
        reply_text=reply_text,
        profile=active_profile,
    )
    if hormone_trace is not None:
        api_pre = pre_snapshot.get("hormones") if isinstance(pre_snapshot, dict) else None
        if api_pre:
            hormone_trace.setdefault("api_pre", dict(api_pre))
        hormone_trace.setdefault("api_post", state_engine.hormone_system.get_state())
    _reinforce_low_self_success(reinforcement, profile=active_profile)
    if telemetry is not None and hormone_trace:
        telemetry["hormone_adjustments"] = hormone_trace
        log_hormone_trace_event(
            hormone_trace,
            telemetry=telemetry,
            reinforcement=reinforcement,
            user=short_user,
            reply=reply_echo,
            intent=intent_prediction.intent,
            length_label=length_plan.get("label"),
            path=HORMONE_TRACE_LOG,
            enabled=HORMONE_TRACE_ENABLED,
            shorten=_shorten,
            logger=logger,
        )
        update_self_narration(
            runtime_state,
            hormone_trace=hormone_trace,
            user_affect=user_affect,
        )
    log_voice_guard_event(
        voice_guard_dict,
        user=short_user,
        reply=reply_echo,
        intent=intent_prediction.intent,
        profile=active_profile,
        path=VOICE_GUARD_LOG,
        shorten=_shorten,
        logger=logger,
    )
    controller_trace = controller_trace_snapshot(runtime_state)
    record_internal_reflection(
        reinforcement,
        reply_text=reply_text,
        intent=intent_prediction,
        state_engine=state_engine,
        runtime_state=runtime_state,
        controller_trace=controller_trace,
        shorten=_shorten,
    )
    model_delta = _update_metric_history(
        reinforcement,
        intent=intent_prediction.intent,
        length_label=length_plan.get("label", ""),
        profile=active_profile,
        pre_hormones=pre_snapshot.get("hormones") if isinstance(pre_snapshot, dict) else None,
    )
    if reinforcement:
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "intent": intent_prediction.intent,
            "source": source,
            "mood": state_engine.state.get("mood", "neutral"),
            "traits": list(state_engine.trait_tags()),
            "metrics": reinforcement,
        }
        if runtime_state.last_metric_averages:
            log_payload["averages"] = dict(runtime_state.last_metric_averages)
        runtime_state.last_reinforcement_metrics = dict(log_payload)
        log_reinforcement_metrics(log_payload, REINFORCEMENT_LOG, logger=logger)
    persona_snapshot = build_persona_snapshot(state_engine)
    apply_persona_feedback(
        persona_snapshot,
        state_engine=state_engine,
        runtime_state=runtime_state,
        hormone_model=HORMONE_MODEL,
    )
    adjusted_state = state_engine.get_state()
    adjusted_persona = build_persona_snapshot(state_engine)
    adjusted_state["persona"] = adjusted_persona
    post_hormones = state_engine.hormone_system.get_state()
    post_working = state_engine.memory_manager.working_snapshot()
    post_summary = _shorten(state_engine.memory_manager.summarize_recent(), 160)
    post_long_term = [
        _shorten(record.content, 160)
        for record in state_engine.memory_manager.recent_long_term(limit=3)
    ]
    api_length_plan = dict(length_plan)
    target_range = api_length_plan.get("target_range")
    if isinstance(target_range, tuple):
        api_length_plan["target_range"] = list(target_range)
    pre_snapshot = telemetry.get("pre", {}) if telemetry else {}
    sampling_snapshot = telemetry.get("sampling") if telemetry else {}
    policy_preview = telemetry.get("policy_preview") if telemetry else None
    llm_metrics = telemetry.get("llm_metrics") if telemetry else None
    controller_snapshot = telemetry.get("controller") if telemetry else None
    controller_input = telemetry.get("controller_input") if telemetry else None
    working_before = pre_snapshot.get("working") if isinstance(pre_snapshot, dict) else []
    turn_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "profile": active_profile,
        "model_alias": telemetry.get("model_alias") if telemetry else LLAMA_MODEL_ALIAS,
        "engine": telemetry.get("engine") if telemetry else source,
        "source": source,
        "intent": intent_prediction.intent,
        "intent_confidence": intent_prediction.confidence,
        "length_label": length_plan.get("label"),
        "user": _shorten(user_message, 180),
        "reply": _shorten(reply_text, 220),
        "sampling": sampling_snapshot,
        "policy_preview": policy_preview,
        "pre": {
            "mood": pre_snapshot.get("mood"),
            "hormones": pre_snapshot.get("hormones"),
            "traits": pre_snapshot.get("traits"),
            "memory_summary": pre_snapshot.get("memory_summary"),
            "working": working_before,
            "long_term_preview": pre_snapshot.get("long_term_preview"),
        },
        "post": {
            "mood": adjusted_state.get("mood"),
            "hormones": post_hormones,
            "memory_summary": post_summary,
            "working": post_working,
            "working_new": [entry for entry in post_working if entry not in (working_before or [])],
            "long_term_preview": post_long_term,
        },
        "reinforcement": reinforcement,
        "model_delta": model_delta,
    }
    if hormone_trace:
        turn_log["hormone_adjustments"] = hormone_trace
    if telemetry and telemetry.get("hormone_style_hits"):
        turn_log["hormone_style_hits"] = telemetry["hormone_style_hits"]
    if telemetry and telemetry.get("affect_overrides"):
        turn_log["affect_overrides"] = telemetry["affect_overrides"]
    if user_affect:
        turn_log["user_affect"] = user_affect.as_dict()
    if llm_metrics:
        turn_log["llm_metrics"] = llm_metrics
    if controller_snapshot:
        turn_log["controller"] = controller_snapshot
    if controller_input:
        turn_log["controller_input"] = controller_input
    if llm_payload and isinstance(llm_payload, dict):
        usage = llm_payload.get("usage")
        if usage:
            turn_log["llm_usage"] = usage
    log_endocrine_turn(turn_log, ENDOCRINE_LOG, logger=logger)
    response: dict[str, Any] = {
        "reply": reply_text,
        "source": source,
        "state": adjusted_state,
        "persona": adjusted_persona,
        "intent": intent_prediction.intent,
        "intent_confidence": intent_prediction.confidence,
        "length_plan": api_length_plan,
        "reinforcement": reinforcement,
    }
    if telemetry and telemetry.get("controller"):
        response["controller"] = telemetry["controller"]
    if telemetry and telemetry.get("controller_input"):
        response["controller_input"] = telemetry["controller_input"]
    if telemetry:
        response["telemetry"] = telemetry
    response["voice_guard"] = voice_guard_dict
    if user_affect:
        response["user_affect"] = user_affect.as_dict()
    if llm_payload is not None:
        response["llm"] = llm_payload
    inner_refs = state_engine.memory_manager.recent_internal_reflections(limit=3)
    if inner_refs:
        response["inner_reflections"] = inner_refs
    if AFFECT_DEBUG_PANEL_ENABLED and runtime_state.last_sampling_snapshot:
        response["sampling_snapshot"] = dict(runtime_state.last_sampling_snapshot)
    if runtime_state.last_reinforcement_metrics:
        response["reinforcement_metrics"] = dict(runtime_state.last_reinforcement_metrics)
    log_webui_interaction(
        user=short_user,
        reply=reply_echo,
        intent=intent_prediction.intent,
        profile=active_profile,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
        path=WEBUI_INTERACTION_LOG,
        pretty_path=WEBUI_INTERACTION_PRETTY_LOG,
        shorten=_shorten,
        logger=logger,
    )
    write_telemetry_snapshot(
        telemetry,
        TELEMETRY_SNAPSHOT_PATH,
        logger=logger,
    )
    collect_persona_sample(
        user=short_user,
        reply=reply_echo,
        reinforcement=reinforcement,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
        destination=PERSONA_SAMPLE_LOG,
        logger=logger,
    )
    collect_helper_sample(
        user=short_user,
        reply=reply_echo,
        reinforcement=reinforcement,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
        destination=HELPER_SAMPLE_LOG,
        logger=logger,
    )
    return response

@app.get("/", response_class=HTMLResponse)
async def serve_chat(request: Request) -> HTMLResponse:
    """Serve the chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Simple health check endpoint reporting the latest mood."""
    return {"status": "alive", "mood": state_engine.state["mood"]}


@app.get("/state")
async def get_state() -> dict[str, Any]:
    """Expose the most recent state snapshot."""
    state = state_engine.snapshot()
    state["persona"] = build_persona_snapshot(state_engine)
    if AFFECT_DEBUG_PANEL_ENABLED:
        debug_payload = {
            "sampling_snapshot": dict(runtime_state.last_sampling_snapshot),
            "affect": state_engine.affect_overview(),
            "reinforcement_metrics": dict(runtime_state.last_reinforcement_metrics),
            "metric_averages": dict(runtime_state.last_metric_averages),
            "inner_reflections": state_engine.memory_manager.recent_internal_reflections(limit=3),
        }
        state["debug"] = debug_payload
    return state


@app.get("/memories")
async def get_recent_memories(limit: int = 5) -> dict[str, Any]:
    """Return recent short-term summary and persisted long-term memories."""
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be positive")
    records = state_engine.memory_manager.recent_long_term(limit=limit)
    return {
        "summary": state_engine.memory_manager.summarize_recent(),
        "working": state_engine.memory_manager.working_snapshot(),
        "long_term": [record.model_dump() for record in records],
    }


@app.post("/events", status_code=202)
async def post_event(payload: EventPayload) -> dict[str, Any]:
    """Record an external interaction and update hormone levels accordingly."""
    try:
        await state_engine.register_event(
            payload.content,
            strength=payload.strength,
            stimulus_type=payload.stimulus,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "accepted", "mood": state_engine.state.get("mood")}


@app.post("/admin/reload")
async def admin_reload() -> dict[str, Any]:
    """Reload configuration and restart LLM backends."""
    summary = await _reload_runtime_settings()
    return summary


@app.get("/admin/caps")
async def admin_caps() -> dict[str, Any]:
    """Report configured token caps and truncation safeguards."""
    return {
        "llama_completion_tokens": LLAMA_COMPLETION_TOKENS,
        "length_overrides": {
            label: overrides.get("max_tokens")
            for label, overrides in LENGTH_SAMPLING_OVERRIDES.items()
        },
        "timeouts": {
            "llm_timeout": LLM_TIMEOUT,
            "llama_server_timeout": LLAMA_SERVER_TIMEOUT,
            "llama_ready_timeout": LLAMA_READINESS_TIMEOUT,
        },
        "base_sampling": {
            "temperature": BASE_TEMPERATURE,
            "top_p": BASE_TOP_P,
            "frequency_penalty": BASE_FREQUENCY_PENALTY,
        },
        "chat_schema": {
            "max_message_chars": 2000,
        },
    }


@app.post("/chat/stream")
async def chat_stream(payload: ChatMessage) -> StreamingResponse:
    """Stream chat responses token-by-token when supported."""
    user_affect = _classify_user_affect(payload.message)
    try:
        await state_engine.register_event(
            f"user: {payload.message}",
            strength=0.7,
            stimulus_type=payload.stimulus,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    (
        context,
        intent_prediction,
        length_plan,
        sampling,
        llm_context,
        snapshot,
    ) = _prepare_chat_request(payload.message, user_affect=user_affect)
    base_telemetry = compose_turn_telemetry(
        context=context,
        sampling=sampling,
        snapshot=snapshot,
        state_engine=state_engine,
        shorten=_shorten,
        model_alias=LLAMA_MODEL_ALIAS,
    )
    if user_affect:
        base_telemetry["user_affect"] = user_affect.as_dict()

    async def event_generator() -> AsyncIterator[str]:
        api_length_plan = dict(length_plan)
        target_range = api_length_plan.get("target_range")
        if isinstance(target_range, tuple):
            api_length_plan["target_range"] = list(target_range)
        init_payload = {
            "intent": intent_prediction.intent,
            "intent_confidence": intent_prediction.confidence,
            "length_plan": api_length_plan,
        }
        yield _sse_event("init", init_payload)
        telemetry_data = dict(base_telemetry)
        reply_text = ""
        source = "heuristic"
        llm_payload: dict[str, Any] | None = None
        active_intent = intent_prediction
        active_length_plan = length_plan
        if local_llama_engine is not None:
            try:
                stream = await local_llama_engine.stream_reply(
                    payload.message, llm_context, sampling=sampling
                )
                async for chunk in stream:
                    kind = chunk.get("type")
                    if kind == "token":
                        token_text = chunk.get("text", "")
                        if token_text:
                            reply_text += token_text
                            yield _sse_event("token", {"text": token_text})
                    elif kind == "done":
                        reply_text = chunk.get("text", reply_text)
                        llm_payload = chunk.get("payload")
                        source = "local"
                        telemetry_data["engine"] = "local"
                        if llm_payload and isinstance(llm_payload, dict):
                            timings = llm_payload.get("timings")
                            if timings:
                                telemetry_data["llm_metrics"] = timings
                        break
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("Local llama streaming failed; falling back: %s", exc)
                reply_text = ""
        if not reply_text:
            (
                reply_text,
                source,
                llm_payload,
                fallback_intent,
                fallback_length_plan,
                telemetry,
            ) = await _generate_chat_reply(payload.message, user_affect=user_affect)
            active_intent = fallback_intent
            active_length_plan = fallback_length_plan
            if reply_text:
                yield _sse_event("token", {"text": reply_text})
            telemetry_data = telemetry
        final_response = await _finalize_chat_response(
            payload.message,
            reply_text,
            source,
            llm_payload,
            active_intent,
            active_length_plan,
            telemetry_data,
            user_affect=user_affect,
        )
        yield _sse_event("complete", final_response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat")
async def chat(payload: ChatMessage) -> dict[str, Any]:
    """Handle chat messages via the living_llm bridge with heuristic fallback."""
    user_affect = _classify_user_affect(payload.message)
    try:
        shared_echo = _shorten(payload.message, 160)
        await state_engine.register_event(
            f"Heard you share {shared_echo}",
            strength=0.7,
            stimulus_type=payload.stimulus,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    (
        reply_text,
        source,
        llm_payload,
        intent_prediction,
        length_plan,
        telemetry,
    ) = await _generate_chat_reply(payload.message, user_affect=user_affect)
    return await _finalize_chat_response(
        payload.message,
        reply_text,
        source,
        llm_payload,
        intent_prediction,
        length_plan,
        telemetry,
        user_affect=user_affect,
    )


__all__ = ["app", "state_engine"]
