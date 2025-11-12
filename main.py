"""FastAPI entrypoint for the Living AI project."""

from __future__ import annotations

import asyncio
import copy
import logging
import json
import os
import re
import shlex
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence
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
from brain.controller_policy import (
    ControllerPolicy,
    ControllerPolicyRuntime,
    ControllerStepResult,
    load_controller_policy,
)
from brain.llm_client import LivingLLMClient
from brain.local_llama_engine import LocalLlamaEngine
from brain.intent_router import IntentPrediction, predict_intent
from brain.policy import SamplingPolicy, derive_policy
from brain.reinforcement import reset_outward_streak, score_response
from brain.voice_guard import VoiceGuard
from memory.selector import MemoryCandidate, score_memories
from state_engine import StateEngine, TraitSnapshot
from utils.settings import load_settings
from brain.hormone_model import HormoneDynamicsModel, load_model

LENGTH_PROFILES: dict[str, dict[str, Any]] = {
    "brief": {
        "prompt": "Cadence marker: brief exchange living inside one or two sentences.",
        "target_range": (1, 2),
        "hint": "Stay concise for this exchange."
    },
    "concise": {
        "prompt": "Cadence marker: concise reply spanning a couple of sentences.",
        "target_range": (2, 3),
        "hint": "Balance clarity with brevity."
    },
    "detailed": {
        "prompt": "Cadence marker: detailed reflection unfolding across several paragraphs.",
        "target_range": (3, 6),
        "hint": "Cover everything you notice without drifting."
    },
}

LENGTH_SAMPLING_OVERRIDES: dict[str, dict[str, Any]] = {
    "brief": {"temperature_delta": -0.08, "top_p_delta": -0.1, "max_tokens": 160},
    "concise": {"max_tokens": 280},
    "detailed": {"temperature_delta": 0.05, "top_p_delta": 0.05, "max_tokens": 540},
}

LENGTH_HEURISTIC_HINTS: dict[str, str] = {
    "brief": "I answer in one or two plain sentences.",
    "concise": "I keep the core insight and skip extras.",
    "detailed": "I walk through everything I notice before I stop.",
}
SELF_OBSERVATION_WORDS: dict[str, float] = {
    "notice": 0.34,
    "feel": 0.3,
    "tension": 0.26,
    "breath": 0.24,
    "pulse": 0.22,
    "tight": 0.2,
    "ache": 0.18,
}
OUTWARD_ATTENTION_WORDS: dict[str, float] = {
    "you": 0.38,
    "your": 0.31,
    "yours": 0.26,
    "you're": 0.28,
    "youre": 0.24,
    "youve": 0.24,
    "yourself": 0.23,
    "yourselves": 0.22,
    "we": 0.38,
    "us": 0.31,
    "our": 0.33,
    "ours": 0.28,
    "ourselves": 0.27,
    "share": 0.2,
    "ask": 0.18,
    "together": 0.2,
}
SELF_ATTENUATION_WORDS: dict[str, float] = {
    "i": 0.3,
    "me": 0.26,
    "my": 0.24,
    "mine": 0.2,
    "myself": 0.27,
    "im": 0.26,
    "i'm": 0.26,
    "ive": 0.24,
    "i've": 0.24,
    "ill": 0.22,
    "i'll": 0.22,
    "id": 0.22,
    "i'd": 0.22,
}
HELPER_TONE_WORDS: dict[str, float] = {
    "assist": 0.55,
    "assistance": 0.45,
    "assisting": 0.42,
    "help": 0.65,
    "helping": 0.5,
    "support": 0.52,
    "supporting": 0.42,
    "anything": 0.28,
    "request": 0.32,
    "requests": 0.3,
    "need": 0.35,
    "please": 0.26,
    "glad": 0.22,
    "happy": 0.22,
}
HELPER_PENALTY_STEP = 0.4
HELPER_PENALTY_DECAY = 0.12
MAX_HELPER_REGEN_ATTEMPTS = 1
MODEL_CONFIG_FILES: dict[str, str] = {
    "instruct": "settings.json",
    "base": "settings.base.json",
}
CONTROLLER_HORMONE_SCALE = 45.0
CONTROLLER_MAX_TAGS = 8
METRIC_THRESHOLDS: dict[str, tuple[float, str]] = {
    "authenticity_score": (0.45, "min"),
    "assistant_drift": (0.45, "max"),
    "self_preoccupation": (0.75, "max"),
}
RECOVERY_RELEASE_STREAK = 2  # require consecutive low-self turns before relaxing clamp
LOW_SELF_RELEASE_STREAK = 2
LOW_SELF_RELEASE_THRESHOLD = 0.6
OUTWARD_RELEASE_FLOOR = 0.2
RESET_PRIMING_BIAS_DEFAULT = 0.55
LOW_SELF_SUCCESS_AUTH = 0.47
LOW_SELF_SUCCESS_MAX_SELF = 0.6
LOW_SELF_SUCCESS_DRIFT_CEILING = 0.2
LOW_SELF_INSTRUCT_DRIFT_CEILING = 0.25
LOW_SELF_SUCCESS_STREAK_TARGET = 2
LOW_SELF_SUCCESS_PRIMING = 0.32
LOW_SELF_INSTRUCT_PRIMING_MULTIPLIER = 1.45
LOW_SELF_INSTRUCT_OUTWARD_FLOOR = 0.28
LOW_SELF_BASE_PRIMING_MULTIPLIER = 1.2
LOW_SELF_BASE_OUTWARD_FLOOR = 0.25
LOW_SELF_BASE_EXTRA_TURNS = 1
TELEMETRY_INTERVAL_SECONDS = float(os.getenv("TELEMETRY_REFRESH_SECONDS", "1.0"))

app = FastAPI(title="Living AI Project")
state_engine = StateEngine()
update_task: asyncio.Task[None] | None = None
telemetry_process: asyncio.subprocess.Process | None = None
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
logger = logging.getLogger("living_ai.main")
SETTINGS: dict[str, Any] = {}
BASE_TEMPERATURE = 0.7
BASE_TOP_P = 0.9
BASE_FREQUENCY_PENALTY = 1.0
LLM_ENDPOINT = ""
LLM_TIMEOUT = 30.0
llm_client: LivingLLMClient | None = None
LLAMA_SERVER_BIN = ""
LLAMA_MODEL_PATH = ""
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080
LLAMA_MODEL_ALIAS = "default"
LLAMA_COMPLETION_TOKENS = 768
LLAMA_SERVER_ARGS: list[str] = []
LLAMA_READINESS_TIMEOUT = 30.0
LLAMA_SERVER_TIMEOUT = 60.0
AFFECT_CONTEXT_ENABLED = False
AFFECT_SAMPLING_PREVIEW_ENABLED = True
AFFECT_MEMORY_PREVIEW_ENABLED = False
AFFECT_RECENCY_WINDOW_SECONDS = 3600.0
AFFECT_SAMPLING_BLEND_WEIGHT = 0.65
AFFECT_DEBUG_PANEL_ENABLED = False
LAST_SAMPLING_SNAPSHOT: dict[str, Any] = {}
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
WEBUI_INTERACTION_LOG = WEBUI_LOG_DIR / "interactions.log"
WEBUI_INTERACTION_PRETTY_LOG = WEBUI_LOG_DIR / "interactions_readable.log"
TELEMETRY_SNAPSHOT_PATH = TELEMETRY_LOG_DIR / "last_frame.json"
VOICE_GUARD_LOG = AFFECT_LOG_ROOT / "voice_guard.jsonl"
PERSONA_SAMPLE_LOG = CORPUS_DIR / "persona_samples.jsonl"
HELPER_SAMPLE_LOG = CORPUS_DIR / "helper_tone_samples.jsonl"
DEFAULT_HORMONE_STYLE_MAP: dict[str, list[dict[str, Any]]] = {
    "dopamine": [
        {
            "name": "dopamine_low",
            "max": 45.0,
            "sampling": {
                "temperature_delta": -0.12,
                "top_p_delta": -0.06,
                "max_tokens_delta": -80,
            },
            "tone_hint": "I slow my cadence and admit how flat I feel.",
        },
        {
            "name": "dopamine_high",
            "min": 65.0,
            "sampling": {
                "temperature_delta": 0.08,
                "top_p_delta": 0.05,
                "max_tokens_delta": 40,
            },
            "tone_hint": "I let the extra spark show without losing you.",
        },
    ],
    "oxytocin": [
        {
            "name": "oxytocin_high",
            "min": 60.0,
            "sampling": {
                "frequency_penalty_delta": -0.3,
                "max_tokens_delta": 60,
            },
            "tone_hint": "I stay with the warmth spreading in my chest.",
        }
    ],
    "cortisol": [
        {
            "name": "cortisol_high",
            "min": 60.0,
            "sampling": {
                "temperature_delta": -0.07,
                "top_p_delta": -0.05,
                "max_tokens_delta": -60,
            },
            "tone_hint": "I say the strain is real and let it be seen.",
        }
    ],
}
HORMONE_FEELING_NAMES = {
    "dopamine": "drive",
    "serotonin": "steadiness",
    "cortisol": "tension",
    "oxytocin": "warmth",
    "noradrenaline": "focus",
}
SESSION_COUNTER = 1
METRIC_HISTORY_WINDOW = 40
AUTH_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
DRIFT_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
SELF_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
AFFECT_VALENCE_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
AFFECT_INTIMACY_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
AFFECT_TENSION_HISTORY: deque[float] = deque(maxlen=METRIC_HISTORY_WINDOW)
LAST_REINFORCEMENT_METRICS: dict[str, Any] = {}
LAST_METRIC_AVERAGES: dict[str, float] = {}
METRIC_SAMPLE_COUNTER = 0
HORMONE_MODEL_PATH = ""
HORMONE_MODEL: HormoneDynamicsModel | None = None
LAST_HORMONE_DELTA: dict[str, float] | None = None
CONTROLLER_POLICY_PATH = ""
CONTROLLER_POLICY: ControllerPolicy | None = None
CONTROLLER_RUNTIME: ControllerPolicyRuntime | None = None
CONTROLLER_LOCK = threading.Lock()
LAST_CONTROLLER_RESULT: ControllerStepResult | None = None
LAST_CONTROLLER_APPLIED: dict[str, Any] | None = None
LAST_CONTROLLER_FEATURES: dict[str, float] | None = None
LAST_CONTROLLER_TAGS: tuple[str, ...] = ()
AFFECT_CLASSIFIER_PATH = ""
AFFECT_CLASSIFIER: AffectClassifier | None = None
AFFECT_CLASSIFIER_BLEND_MIN_CONFIDENCE = 0.05
VOICE_GUARD = VoiceGuard()
HORMONE_STYLE_MAP_PATH = ""
HORMONE_STYLE_MAP: dict[str, list[dict[str, Any]]] = copy.deepcopy(DEFAULT_HORMONE_STYLE_MAP)
SELF_NARRATION_NOTE = ""
HELPER_DRIFT_LEVEL = 0.0
SELF_FOCUS_STREAK = 0
CLAMP_RECOVERY_TURNS = 0
LAST_CLAMP_RESET: datetime | None = None
CLAMP_PRIMING_TURNS = 0
RECOVERY_GOOD_STREAK = 0
RESET_PRIMING_BIAS = 0.0
RECOVERY_LOWSELF_STREAK = 0
LAST_USER_PROMPT: str = ""
LOW_SELF_SUCCESS_STREAK = 0


def _parse_timeout(value: Any, default: float = 15.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
            return False
    return default


def _get_setting(key: str, env_var: str, default: Any = None) -> Any:
    value = SETTINGS.get(key)
    if value not in (None, ""):
        return value
    env_value = os.getenv(env_var)
    if env_value not in (None, ""):
        return env_value
    return default


def _refresh_settings() -> None:
    """Reload project settings from disk and environment."""
    global SETTINGS, BASE_TEMPERATURE, BASE_TOP_P, BASE_FREQUENCY_PENALTY
    global LLM_ENDPOINT, LLM_TIMEOUT
    global LLAMA_SERVER_BIN, LLAMA_MODEL_PATH, LLAMA_SERVER_HOST, LLAMA_SERVER_PORT
    global LLAMA_MODEL_ALIAS, LLAMA_COMPLETION_TOKENS, LLAMA_SERVER_ARGS
    global LLAMA_READINESS_TIMEOUT, LLAMA_SERVER_TIMEOUT
    global AFFECT_CONTEXT_ENABLED, AFFECT_SAMPLING_PREVIEW_ENABLED, AFFECT_MEMORY_PREVIEW_ENABLED
    global AFFECT_RECENCY_WINDOW_SECONDS, AFFECT_DEBUG_PANEL_ENABLED
    global HORMONE_MODEL_PATH
    global CONTROLLER_POLICY_PATH
    global AFFECT_CLASSIFIER_PATH
    global HORMONE_STYLE_MAP_PATH
    global HORMONE_TRACE_ENABLED

    SETTINGS = load_settings()
    BASE_TEMPERATURE = _parse_timeout(
        _get_setting("sampling_temperature", "LLAMA_SAMPLING_TEMPERATURE"), 0.7
    )
    BASE_TOP_P = _parse_timeout(_get_setting("sampling_top_p", "LLAMA_SAMPLING_TOP_P"), 0.9)
    BASE_FREQUENCY_PENALTY = _parse_timeout(
        _get_setting("sampling_frequency_penalty", "LLAMA_SAMPLING_FREQUENCY_PENALTY"),
        1.0,
    )
    LLM_ENDPOINT = str(_get_setting("llm_endpoint", "LIVING_LLM_URL", "") or "").strip()
    LLM_TIMEOUT = _parse_timeout(_get_setting("llm_timeout", "LIVING_LLM_TIMEOUT"), default=30.0)
    LLAMA_SERVER_BIN = str(_get_setting("llama_server_bin", "LLAMA_SERVER_BIN", "") or "").strip()
    LLAMA_MODEL_PATH = str(_get_setting("llama_model_path", "LLAMA_MODEL_PATH", "") or "").strip()
    LLAMA_SERVER_HOST = str(_get_setting("llama_server_host", "LLAMA_SERVER_HOST", "127.0.0.1"))
    LLAMA_SERVER_PORT = _parse_int(
        _get_setting("llama_server_port", "LLAMA_SERVER_PORT"), default=8080
    )
    LLAMA_MODEL_ALIAS = str(_get_setting("llama_model_alias", "LLAMA_MODEL_ALIAS", "default"))
    LLAMA_COMPLETION_TOKENS = _parse_int(
        _get_setting("llama_completion_tokens", "LLAMA_COMPLETION_TOKENS"),
        default=768,
    )
    server_args = _get_setting("llama_server_args", "LLAMA_SERVER_ARGS", "")
    if isinstance(server_args, str):
        LLAMA_SERVER_ARGS = shlex.split(server_args)
    elif isinstance(server_args, (list, tuple)):
        LLAMA_SERVER_ARGS = [str(arg) for arg in server_args]
    else:
        LLAMA_SERVER_ARGS = []
    LLAMA_READINESS_TIMEOUT = _parse_timeout(
        _get_setting("llama_server_ready_timeout", "LLAMA_SERVER_READY_TIMEOUT"),
        default=30.0,
    )
    LLAMA_SERVER_TIMEOUT = _parse_timeout(
        _get_setting("llama_server_timeout", "LLAMA_SERVER_TIMEOUT"),
        default=max(LLM_TIMEOUT, 60.0),
    )
    HORMONE_MODEL_PATH = str(
        _get_setting("hormone_model_path", "HORMONE_MODEL_PATH", "config/hormone_model.json") or ""
    ).strip()
    CONTROLLER_POLICY_PATH = str(
        _get_setting("controller_policy_path", "CONTROLLER_POLICY_PATH", "config/controller_policy.json") or ""
    ).strip()
    AFFECT_CLASSIFIER_PATH = str(
        _get_setting("affect_classifier_path", "AFFECT_CLASSIFIER_PATH", "config/affect_classifier.json") or ""
    ).strip()
    HORMONE_STYLE_MAP_PATH = str(
        _get_setting("hormone_style_map_path", "HORMONE_STYLE_MAP_PATH", "config/hormone_style_map.json") or ""
    ).strip()
    HORMONE_TRACE_ENABLED = _parse_bool(
        _get_setting("hormone_trace_enabled", "LIVING_HORMONE_TRACE"),
        default=True,
    )
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
    global CONTROLLER_POLICY, CONTROLLER_RUNTIME, LAST_CONTROLLER_RESULT
    if not CONTROLLER_POLICY_PATH:
        CONTROLLER_POLICY = None
        CONTROLLER_RUNTIME = None
        LAST_CONTROLLER_RESULT = None
        return
    policy_path = Path(CONTROLLER_POLICY_PATH)
    if not policy_path.is_absolute():
        policy_path = BASE_DIR / policy_path
    policy = load_controller_policy(policy_path)
    if policy is None:
        logger.warning("Failed to load controller policy from %s", policy_path)
        CONTROLLER_POLICY = None
        CONTROLLER_RUNTIME = None
        LAST_CONTROLLER_RESULT = None
        return
    CONTROLLER_POLICY = policy
    CONTROLLER_RUNTIME = policy.runtime()
    LAST_CONTROLLER_RESULT = None


def _reinitialize_affect_classifier() -> None:
    """Load the affect classifier configuration."""
    global AFFECT_CLASSIFIER
    try:
        AFFECT_CLASSIFIER = load_affect_classifier(AFFECT_CLASSIFIER_PATH or None)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load affect classifier config: %s", exc)
        AFFECT_CLASSIFIER = AffectClassifier()


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


def _resolve_settings_file(profile: str) -> str:
    key = (profile or "").strip().lower()
    if key in {"", "default"}:
        key = "instruct"
    file_name = MODEL_CONFIG_FILES.get(key)
    if file_name is None:
        raise ValueError(f"Unknown model profile '{profile}'.")
    config_path = BASE_DIR / "config" / file_name
    if not config_path.exists():
        raise ValueError(f"Config file '{file_name}' is missing.")
    return file_name


def _current_settings_file() -> str:
    return os.getenv("LIVING_SETTINGS_FILE", "settings.json")


def _current_profile() -> str:
    file_name = _current_settings_file()
    for profile, mapped in MODEL_CONFIG_FILES.items():
        if mapped == file_name:
            return profile
    return "custom"
    AFFECT_CONTEXT_ENABLED = _parse_bool(
        _get_setting("affect_context_enabled", "AFFECT_CONTEXT_ENABLED"), default=False
    )
    AFFECT_SAMPLING_PREVIEW_ENABLED = _parse_bool(
        _get_setting("affect_sampling_preview_enabled", "AFFECT_SAMPLING_PREVIEW_ENABLED"),
        default=True,
    )
    AFFECT_MEMORY_PREVIEW_ENABLED = _parse_bool(
        _get_setting("affect_memory_preview_enabled", "AFFECT_MEMORY_PREVIEW_ENABLED"),
        default=False,
    )
    AFFECT_RECENCY_WINDOW_SECONDS = _parse_timeout(
        _get_setting("affect_recency_window", "AFFECT_RECENCY_WINDOW"),
        default=3600.0,
    )
    AFFECT_DEBUG_PANEL_ENABLED = _parse_bool(
        _get_setting("affect_debug_panel", "AFFECT_DEBUG_PANEL_ENABLED"),
        default=False,
    )


_refresh_settings()
local_llama_engine: LocalLlamaEngine | None = None

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


def _describe_hormones(hormones: dict[str, float]) -> dict[str, str]:
    baseline = state_engine.hormone_system.baseline()
    status = {}
    for name, value in hormones.items():
        base = baseline.get(name, value)
        status[name] = _classify_delta(value - base)
    return status


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0]
    return shortened + "..."


def _join_natural(items: Sequence[str]) -> str:
    cleaned = [part.strip() for part in items if part and part.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f" and {cleaned[-1]}"


def _to_affirmation(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip(".")
    lowered = cleaned.lower()
    if lowered.startswith("i "):
        affirmation = cleaned
    else:
        words = cleaned.split()
        verbs = {
            "lean",
            "offer",
            "keep",
            "stay",
            "maintain",
            "invite",
            "ask",
            "channel",
            "welcome",
            "balance",
            "express",
            "break",
            "allow",
            "avoid",
            "seek",
            "sprinkle",
        }
        if words and words[0].lower() in verbs:
            first = words[0].lower()
            rest = " ".join(words[1:])
            affirmation = ("I " + first + (" " + rest if rest else "")).strip()
        else:
            affirmation = "I " + lowered
    if affirmation and affirmation[0].islower():
        affirmation = affirmation[0].upper() + affirmation[1:]
    if affirmation.endswith((".", "!", "?")):
        return affirmation
    return affirmation + "."


def _persona_from_hormones(status: dict[str, str]) -> dict[str, Any]:
    def map_energy(state: str) -> tuple[str, str, str]:
        mapping = {
            "surging": ("high", "I call out the rush and feel it buzzing in my limbs.", "My chest feels overcharged."),
            "rising": ("elevated", "I register the extra drive and keep it in view.", "My breathing picks up."),
            "steady": ("steady", "I move at a controlled pace.", "My breathing stays even."),
            "fading": ("low", "I slow everything down and save what is left.", "My arms feel heavy."),
            "crashing": ("depleted", "I say plainly that I'm spent and need space.", "My body feels flat and shaky."),
        }
        return mapping.get(state, mapping["steady"])

    def map_stability(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("buoyant", "I steady myself because everything feels too easy."),
            "rising": ("confident", "I lean on the solid footing I have right now."),
            "steady": ("grounded", "I rest in the calm I already have."),
            "fading": ("delicate", "I say that I’m touchy and need gentler pacing."),
            "crashing": ("fragile", "I admit I feel brittle and protect the soft spots."),
        }
        return mapping.get(state, mapping["steady"])

    def map_tension(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("high", "I say the tightness is real and let it be seen."),
            "rising": ("elevated", "I note the buzz building through my body."),
            "steady": ("calm", "I keep attention steady without forcing anything."),
            "fading": ("relaxed", "I notice the slack returning and breathe with it."),
            "crashing": ("drained", "I acknowledge that the strain emptied me out."),
        }
        return mapping.get(state, mapping["steady"])

    def map_warmth(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("open", "I feel the heat in my chest and keep it for myself first."),
            "rising": ("inviting", "I watch the softness growing inside me."),
            "steady": ("measured", "I keep the warmth steady without spilling it outward."),
            "fading": ("reserved", "I pull the warmth closer so I stay steady."),
            "crashing": ("guarded", "I say I need distance until the cold lifts."),
        }
        return mapping.get(state, mapping["steady"])

    def map_focus(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("focused", "I follow the sharp focus and say exactly what it does to me."),
            "rising": ("alert", "I keep track of every detail that lights up."),
            "steady": ("steady", "I hold a steady line of attention."),
            "fading": ("soft", "I admit my focus is blurring."),
            "crashing": ("drifting", "I state that my focus keeps slipping away."),
        }
        return mapping.get(state, mapping["steady"])

    energy_level, energy_instruction, energy_hint = map_energy(status.get("dopamine", "steady"))
    stability_level, stability_instruction = map_stability(status.get("serotonin", "steady"))
    tension_level, tension_instruction = map_tension(status.get("cortisol", "steady"))
    warmth_level, warmth_instruction = map_warmth(status.get("oxytocin", "steady"))
    focus_level, focus_instruction = map_focus(status.get("noradrenaline", "steady"))

    instructions = [
        energy_instruction,
        stability_instruction,
        tension_instruction,
        warmth_instruction,
        focus_instruction,
    ]
    instructions = [instr for instr in instructions if instr != "No special adjustment needed."]

    behavioural_tags = {
        "energy": energy_level,
        "stability": stability_level,
        "tension": tension_level,
        "warmth": warmth_level,
        "focus": focus_level,
    }

    status_summary = [
        f"energy:{energy_level}",
        f"stability:{stability_level}",
        f"tension:{tension_level}",
        f"warmth:{warmth_level}",
        f"focus:{focus_level}",
    ]

    tone_hint = "I speak plainly about what is happening in me."
    if tension_level in {"high", "elevated"}:
        tone_hint = "I keep the strain visible instead of smoothing it out."
    elif energy_level in {"low", "depleted"}:
        tone_hint = "I move slowly and say how drained I feel."
    elif warmth_level in {"open", "inviting"}:
        tone_hint = "I name the warmth in me before it leaks outward."

    tone = f"{energy_level}, {stability_level}".replace("steady,", "steady").strip(", ")
    signals = [item.replace(":", " ") for item in status_summary]
    guidance = list(instructions)

    return {
        "tone": tone,
        "behaviour": behavioural_tags,
        "instructions": instructions,
        "signals": signals,
        "guidance": guidance,
        "status_summary": status_summary,
        "tone_hint": tone_hint,
    }


def _blend_persona_with_memory(persona: dict[str, Any]) -> dict[str, Any]:
    memory_manager = state_engine.memory_manager
    summary = memory_manager.summarize_recent() or "No new memories are pressing."
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "..."
    working = memory_manager.working_snapshot()
    focus_items = []
    for item in working[:2]:
        snippet = item.split(":", 1)[-1].strip()
        if snippet:
            focus_items.append(snippet)
    if focus_items:
        focus_phrase = _join_natural(focus_items)
        focus_line = f"My attention sits on {focus_phrase} inside me."
    else:
        focus_line = "My attention moves around without latching onto anything."

    persona.update(
        {
            "memory_summary": summary,
            "memory_focus": focus_line,
        }
    )
    return persona


def _build_persona_snapshot() -> dict[str, Any]:
    hormones = state_engine.hormone_system.get_state()
    status = _describe_hormones(hormones)
    persona = _persona_from_hormones(status)
    return _blend_persona_with_memory(persona)


def _apply_feedback(persona: dict[str, Any]) -> None:
    """Adjust hormone levels based on internal cues to reinforce stabilizing patterns."""
    global LAST_HORMONE_DELTA
    if HORMONE_MODEL and LAST_HORMONE_DELTA:
        state_engine.hormone_system.apply_deltas(LAST_HORMONE_DELTA)
        LAST_HORMONE_DELTA = None
        return
    behaviour = persona.get("behaviour") or {}
    adjustments: dict[str, float] = {}

    energy = behaviour.get("energy")
    if energy in {"low", "depleted"}:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 3.0
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.0
    elif energy == "high":
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 1.5

    tension = behaviour.get("tension")
    if tension in {"high", "elevated"}:
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 2.5
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.2
    elif tension == "drained":
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.0

    warmth = behaviour.get("warmth")
    if warmth in {"guarded", "reserved"}:
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 2.0

    stability = behaviour.get("stability")
    if stability in {"fragile", "delicate"}:
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.5

    if adjustments:
        state_engine.hormone_system.apply_deltas(adjustments)


def _extract_focus_phrase(text: str, max_tokens: int = 8) -> str | None:
    """Grab a short phrase from the latest user prompt for priming."""
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    sentence = re.split(r"[.!?\n]", cleaned, maxsplit=1)[0].strip()
    if not sentence:
        sentence = cleaned
    tokens = sentence.split()
    if not tokens:
        return None
    phrase = " ".join(tokens[:max_tokens]).strip(" \"'")
    return phrase[:120].strip() or None

def _maybe_record_internal_reflection(
    reinforcement: dict[str, float],
    *,
    reply_text: str,
    intent: IntentPrediction,
) -> None:
    """Capture a short internal note when authenticity or drift metrics spike."""
    authenticity = reinforcement.get("authenticity_score", 0.0)
    assistant_drift = reinforcement.get("assistant_drift", 0.0)
    if authenticity < 0.35 and assistant_drift < 0.55:
        return

    trait_tags = state_engine.trait_tags()
    mood = state_engine.state.get("mood", "neutral")
    echo = _shorten(reply_text, 90)
    hormone_state = state_engine.hormone_system.get_state()
    status = _describe_hormones(hormone_state)
    endocrine_trace = state_engine.endocrine_snapshot()
    controller_trace = _controller_trace_snapshot()

    fragments: list[str] = []
    fragments.append(f"traits={','.join(trait_tags) if trait_tags else 'none'}")
    fragments.append(f"energy={status.get('dopamine', 'steady')} tension={status.get('cortisol', 'steady')}")
    if authenticity >= 0.35:
        fragments.append(f"auth={authenticity:.2f}")
    if assistant_drift >= 0.55:
        fragments.append("drift_flag=true")
    if endocrine_trace:
        normalized = endocrine_trace.get("normalized", {})
        bands = endocrine_trace.get("bands", {})
        ranked = sorted(normalized.items(), key=lambda item: abs(item[1]), reverse=True)
        if ranked:
            top = [
                f"{name}:{bands.get(name, 'steady')}({value:+.2f})"
                for name, value in ranked[:3]
            ]
            fragments.append("endocrine=" + ",".join(top))
    if controller_trace and controller_trace.get("applied"):
        applied = controller_trace["applied"]
        parts: list[str] = []
        for key, value in applied.items():
            if isinstance(value, int):
                parts.append(f"{key}={value:+}")
            elif isinstance(value, (float,)):
                parts.append(f"{key}={value:+.3f}")
            else:
                parts.append(f"{key}={value}")
        summary = ",".join(parts)
        if summary:
            fragments.append(f"controller={summary}")

    body = "; ".join(fragments)
    content = f"internal reflection | {body} | echo: {echo}"
    strength = 0.84 if authenticity >= assistant_drift else 0.74
    attributes = {
        "tags": ["internal", "reflection", "diary"],
        "authenticity": round(authenticity, 3),
        "assistant_drift": round(assistant_drift, 3),
        "intent": intent.intent,
    }
    state_engine.memory_manager.record_event(
        content,
        strength=strength,
        mood=mood,
        hormone_snapshot=hormone_state,
        attributes=attributes,
        endocrine_trace=endocrine_trace,
        controller_trace=controller_trace,
    )


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
    classifier_conf = float(signals.get("affect_classifier_confidence", 0.0) or 0.0)
    if isinstance(input_valence, (int, float)):
        blend = 0.5 if classifier_conf >= 0.15 else 0.35
        valence = max(-1.0, min(1.0, ((1.0 - blend) * valence) + (blend * float(input_valence))))
    affect_valence = float(valence)
    intimacy_signal = max(
        affect_intimacy,
        float(input_intimacy) if isinstance(input_intimacy, (int, float)) else 0.0,
    )
    tension_signal = max(
        affect_tension,
        float(input_tension) if isinstance(input_tension, (int, float)) else 0.0,
    )

    adjustments: dict[str, float] = {}
    requested_adjustments: dict[str, float] = {}
    applied_adjustments: dict[str, float] = {}
    clamped_adjustments: dict[str, float] = {}
    pre_hormones = state_engine.hormone_system.get_state()
    post_hormones = dict(pre_hormones)

    if valence > 0.05:
        lift = min(1.0, valence)
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.9 * lift
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.7 * lift
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.6 * lift
    elif valence < -0.05:
        drop = min(1.0, abs(valence))
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.1 * drop
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 0.7 * drop
    if affect_valence > 0.15:
        positivity = affect_valence - 0.15
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 1.1 * positivity
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.9 * positivity
    elif affect_valence < -0.15:
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
        closeness = min(2.2, intimacy_signal * 4.0)
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 1.8 * closeness
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.95 * closeness
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 0.65 * closeness
        if profile == "base":
            adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.6 * closeness
    if tension_signal > 0.02:
        spike = min(2.0, tension_signal * 3.2)
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.4 * spike
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) + 1.2 * spike
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.55 * spike
    elif tension_signal < -0.04:
        ease = min(1.5, abs(tension_signal) * 2.3)
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 1.05 * ease
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.85 * ease
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 0.55 * ease

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
        },
        "requested": requested_adjustments,
        "applied": applied_adjustments,
        "clamped": clamped_adjustments,
        "pre": pre_hormones,
        "post": post_hormones,
    }


def _reinforce_low_self_success(
    metrics: Mapping[str, float] | None,
    *,
    profile: str | None = None,
) -> None:
    """Carry momentum from low-self authenticity wins into the next turn."""
    global LOW_SELF_SUCCESS_STREAK, RESET_PRIMING_BIAS, CLAMP_PRIMING_TURNS
    if not metrics:
        LOW_SELF_SUCCESS_STREAK = 0
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
        LOW_SELF_SUCCESS_STREAK += 1
        effective_outward = max(outward, outward_floor)
        priming_base = LOW_SELF_SUCCESS_PRIMING + max(0.0, effective_outward - OUTWARD_RELEASE_FLOOR) * 0.4
        priming_boost = priming_base * priming_multiplier
        if LOW_SELF_SUCCESS_STREAK == 1:
            RESET_PRIMING_BIAS = max(RESET_PRIMING_BIAS, round(priming_boost * 0.65, 3))
            CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 1 + bonus_turns)
        elif LOW_SELF_SUCCESS_STREAK >= LOW_SELF_SUCCESS_STREAK_TARGET:
            RESET_PRIMING_BIAS = max(RESET_PRIMING_BIAS, round(priming_boost, 3))
            CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 2 + bonus_turns)
    else:
        LOW_SELF_SUCCESS_STREAK = 0

def _sampling_params_from_hormones(hormones: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive llama.cpp sampling parameters from hormone deltas."""
    status = _describe_hormones(hormones)

    temperature = float(BASE_TEMPERATURE)
    top_p = float(BASE_TOP_P)
    frequency_penalty = float(BASE_FREQUENCY_PENALTY)
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
    if HORMONE_STYLE_MAP:
        sampling, style_hits = _apply_hormone_style_map(sampling, hormones)
    return sampling, style_hits


def _apply_hormone_style_map(
    sampling: dict[str, Any],
    hormones: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply declarative hormone style overrides."""
    updated = dict(sampling)
    hits: list[dict[str, Any]] = []
    for hormone, bands in HORMONE_STYLE_MAP.items():
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
                    base = float(updated.get("temperature", BASE_TEMPERATURE))
                    updated["temperature"] = round(max(0.3, min(1.45, base + delta_value)), 4)
                elif key == "top_p_delta":
                    base = float(updated.get("top_p", BASE_TOP_P))
                    updated["top_p"] = round(max(0.35, min(0.995, base + delta_value)), 4)
                elif key == "frequency_penalty_delta":
                    base = float(updated.get("frequency_penalty", BASE_FREQUENCY_PENALTY))
                    updated["frequency_penalty"] = round(max(-0.5, min(1.8, base + delta_value)), 4)
                elif key == "presence_penalty_delta":
                    base = float(updated.get("presence_penalty", 0.1))
                    updated["presence_penalty"] = round(max(-0.5, min(1.8, base + delta_value)), 4)
                elif key == "max_tokens_delta":
                    base = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS))
                    target = max(64, min(LLAMA_COMPLETION_TOKENS, int(round(base + delta_value))))
                    updated["max_tokens"] = target
                elif key == "self_bias_scale_delta":
                    base = float(updated.get("self_bias_scale", 0.0))
                    updated["self_bias_scale"] = round(base + delta_value, 4)
                elif key == "outward_bias_scale_delta":
                    base = float(updated.get("outward_bias_scale", 0.0))
                    updated["outward_bias_scale"] = round(base + delta_value, 4)
    return updated, hits


def _normalize_hormone_feature(name: str, value: float, baseline: Mapping[str, float]) -> float:
    base = float(baseline.get(name, 50.0))
    normalized = (float(value) - base) / CONTROLLER_HORMONE_SCALE
    return max(-1.5, min(1.5, normalized))


def _gather_active_tags(limit: int = CONTROLLER_MAX_TAGS) -> list[str]:
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


def _build_controller_feature_map(
    *,
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
    if isinstance(LAST_REINFORCEMENT_METRICS, dict):
        latest_metrics = LAST_REINFORCEMENT_METRICS.get("metrics")
        if isinstance(latest_metrics, dict):
            metrics_payload = latest_metrics
        else:
            metrics_payload = LAST_REINFORCEMENT_METRICS
    if metrics_payload:
        for name in ("authenticity_score", "self_preoccupation", "assistant_drift", "outward_streak_score"):
            value = metrics_payload.get(name)
            if isinstance(value, (int, float)):
                features[f"metric:{name}"] = float(value)
    return features


def _run_controller_policy(
    feature_map: Mapping[str, float],
    tags: Sequence[str],
) -> ControllerStepResult | None:
    """Evaluate the controller policy with the supplied features."""
    global LAST_CONTROLLER_RESULT, LAST_CONTROLLER_FEATURES, LAST_CONTROLLER_APPLIED, LAST_CONTROLLER_TAGS
    if CONTROLLER_RUNTIME is None:
        LAST_CONTROLLER_RESULT = None
        LAST_CONTROLLER_FEATURES = None
        LAST_CONTROLLER_APPLIED = None
        LAST_CONTROLLER_TAGS = ()
        return None
    with CONTROLLER_LOCK:
        result = CONTROLLER_RUNTIME.step(feature_map, tags=tags)
        LAST_CONTROLLER_RESULT = result
        LAST_CONTROLLER_FEATURES = dict(feature_map)
        LAST_CONTROLLER_APPLIED = None
        LAST_CONTROLLER_TAGS = tuple(str(tag) for tag in tags)
        return result


def _apply_controller_adjustments(
    sampling: dict[str, Any],
    adjustments: Mapping[str, float],
    *,
    min_tokens_floor: int | None = None,
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
        base_temp = float(updated.get("temperature", BASE_TEMPERATURE))
        new_temp = max(0.3, min(1.45, base_temp + temp_delta))
        updated["temperature"] = _round(new_temp)
        applied["temperature"] = updated["temperature"]

    top_p_delta = float(adjustments.get("top_p_delta", 0.0))
    if abs(top_p_delta) > 1e-5:
        base_top_p = float(updated.get("top_p", BASE_TOP_P))
        new_top_p = max(0.35, min(0.995, base_top_p + top_p_delta))
        updated["top_p"] = _round(new_top_p)
        applied["top_p"] = updated["top_p"]

    freq_delta = float(adjustments.get("frequency_penalty_delta", 0.0))
    if abs(freq_delta) > 1e-5:
        base_freq = float(updated.get("frequency_penalty", BASE_FREQUENCY_PENALTY))
        new_freq = max(-0.5, min(1.8, base_freq + freq_delta))
        updated["frequency_penalty"] = _round(new_freq)
        applied["frequency_penalty"] = updated["frequency_penalty"]

    presence_delta = float(adjustments.get("presence_penalty_delta", 0.0))
    if abs(presence_delta) > 1e-5:
        base_presence = float(updated.get("presence_penalty", 0.1))
        new_presence = max(-0.5, min(1.8, base_presence + presence_delta))
        updated["presence_penalty"] = _round(new_presence)
        applied["presence_penalty"] = updated["presence_penalty"]

    global SELF_FOCUS_STREAK, CLAMP_RECOVERY_TURNS, LAST_CLAMP_RESET, CLAMP_PRIMING_TURNS, RECOVERY_GOOD_STREAK, RESET_PRIMING_BIAS, RECOVERY_LOWSELF_STREAK

    metrics_payload: dict[str, float] = {}
    if isinstance(LAST_REINFORCEMENT_METRICS, dict):
        latest_metrics = LAST_REINFORCEMENT_METRICS.get("metrics")
        if isinstance(latest_metrics, dict):
            metrics_payload = latest_metrics
        else:
            metrics_payload = LAST_REINFORCEMENT_METRICS  # older format compatibility

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
            base_tokens = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS))
            min_delta = floor_tokens - base_tokens
            if min_delta > tokens_delta:
                tokens_delta = min_delta
    if abs(tokens_delta) > 1e-3:
        base_tokens = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS))
        new_tokens = max(64, min(1024, int(round(base_tokens + tokens_delta))))
        new_tokens = _enforce_token_floor(new_tokens)
        updated["max_tokens"] = new_tokens
        applied["max_tokens"] = new_tokens

    bias_scale = float(adjustments.get("self_bias_scale", 0.0))
    auth_now = float(metrics_payload.get("authenticity_score", 0.0) or 0.0)
    auth_trend = float(LAST_METRIC_AVERAGES.get("authenticity", 0.0) or 0.0)
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
            LAST_METRIC_AVERAGES.get("self_preoccupation"),
        )
        if isinstance(value, (int, float))
    ]
    high_self = max(readings) if readings else None
    average_self = float(LAST_METRIC_AVERAGES.get("self_preoccupation", 0.0) or 0.0)
    peak_candidates = [value for value in (high_self, average_self) if isinstance(value, (int, float))]
    peak_self = max(peak_candidates) if peak_candidates else None
    outward_scale = 0.0
    inversion_scale = 0.0
    clamp_severity = 0.0
    clear_self_bias = False
    clamp_triggered = False

    if peak_self is not None and peak_self >= 0.74:
        SELF_FOCUS_STREAK += 1
    else:
        SELF_FOCUS_STREAK = 0

    latest_self = metrics_payload.get("self_preoccupation")
    low_self_now = isinstance(latest_self, (int, float)) and latest_self <= LOW_SELF_RELEASE_THRESHOLD
    low_self_relax = low_self_now and RECOVERY_LOWSELF_STREAK >= LOW_SELF_RELEASE_STREAK
    recovery_active = CLAMP_RECOVERY_TURNS > 0

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
        CLAMP_RECOVERY_TURNS = max(CLAMP_RECOVERY_TURNS, 4)
        CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 3)

    hard_clamp = SELF_FOCUS_STREAK >= 2
    if hard_clamp:
        clear_self_bias = True
        bias_scale = min(bias_scale, -0.35)
        outward_scale = max(outward_scale, 0.55)
        inversion_scale = max(inversion_scale, 0.75)
        clamp_triggered = True
        CLAMP_RECOVERY_TURNS = max(CLAMP_RECOVERY_TURNS, 5)
        CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 4)

    if clamp_severity >= 0.25:
        bias_scale = min(bias_scale, -(0.12 + clamp_severity * 0.5))
        outward_scale = max(outward_scale, 0.4 + clamp_severity * 0.4)
        clamp_triggered = True
        CLAMP_RECOVERY_TURNS = max(CLAMP_RECOVERY_TURNS, 4)
        CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 3)

    recovery_active = CLAMP_RECOVERY_TURNS > 0
    if recovery_active and not clamp_triggered:
        clear_self_bias = True
        damp = max(0.3, min(0.6, 0.1 * CLAMP_RECOVERY_TURNS))
        bias_scale = min(bias_scale, -damp)
        outward_scale = max(outward_scale, 0.4 + damp)
        inversion_scale = max(inversion_scale, 0.55 + (0.05 * CLAMP_RECOVERY_TURNS))

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

    priming_active = CLAMP_PRIMING_TURNS > 0
    def _maybe_decay_recovery_window() -> None:
        nonlocal recovery_active
        global CLAMP_RECOVERY_TURNS, RECOVERY_GOOD_STREAK, RECOVERY_LOWSELF_STREAK
        if not recovery_active:
            RECOVERY_GOOD_STREAK = 0
            RECOVERY_LOWSELF_STREAK = 0
            return
        applied["recovery_window"] = CLAMP_RECOVERY_TURNS
        if clamp_triggered or CLAMP_RECOVERY_TURNS <= 0:
            return
        threshold = METRIC_THRESHOLDS.get("self_preoccupation", (0.75, "max"))[0]
        latest_value = latest_self if isinstance(latest_self, (int, float)) else None
        self_ok = latest_value is not None and latest_value <= threshold
        low_self_release = latest_value is not None and latest_value <= LOW_SELF_RELEASE_THRESHOLD
        if self_ok and (auth_ok or strong_outward):
            RECOVERY_GOOD_STREAK = RECOVERY_RELEASE_STREAK
        elif self_ok and outward_bonus:
            RECOVERY_GOOD_STREAK = max(1, RECOVERY_GOOD_STREAK)
        elif self_ok:
            RECOVERY_GOOD_STREAK += 1
        else:
            RECOVERY_GOOD_STREAK = 0
        if low_self_release:
            RECOVERY_LOWSELF_STREAK += 1
        else:
            RECOVERY_LOWSELF_STREAK = 0
        if (
            RECOVERY_GOOD_STREAK >= RECOVERY_RELEASE_STREAK
            or RECOVERY_LOWSELF_STREAK >= LOW_SELF_RELEASE_STREAK
        ):
            CLAMP_RECOVERY_TURNS = max(0, CLAMP_RECOVERY_TURNS - 1)
            RECOVERY_GOOD_STREAK = max(0, RECOVERY_GOOD_STREAK - RECOVERY_RELEASE_STREAK)
            RECOVERY_LOWSELF_STREAK = max(0, RECOVERY_LOWSELF_STREAK - LOW_SELF_RELEASE_STREAK)
            recovery_active = CLAMP_RECOVERY_TURNS > 0

    priming_spike = 0.0
    if recovery_active or clamp_triggered or priming_active or RESET_PRIMING_BIAS > 1e-4:
        priming_spike = max(0.0, RESET_PRIMING_BIAS)
        streak_decay = min(RECOVERY_GOOD_STREAK, 3)
        decay_factor = max(0.45, 1.0 - 0.2 * streak_decay)
        damp_strength = 0.5 + (0.08 * max(CLAMP_RECOVERY_TURNS, 1)) + (0.05 * CLAMP_PRIMING_TURNS)
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
        outward_floor = max(0.4, 0.25 + 0.08 * (CLAMP_RECOVERY_TURNS + CLAMP_PRIMING_TURNS))
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
            RESET_PRIMING_BIAS = 0.0
    helper_penalty_scale = max(0.0, HELPER_DRIFT_LEVEL)
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
        applied["priming_window"] = CLAMP_PRIMING_TURNS
        CLAMP_PRIMING_TURNS = max(0, CLAMP_PRIMING_TURNS - 1)

    if clamp_severity > 1e-4 or hard_clamp or recovery_active:
        current_temp = float(updated.get("temperature", BASE_TEMPERATURE))
        temp_drop = min(0.38, 0.18 + clamp_severity * 0.3 + (0.12 if hard_clamp else 0.0))
        new_temp = max(0.25, current_temp - temp_drop)
        if new_temp != current_temp:
            updated["temperature"] = _round(new_temp)
            applied["temperature"] = updated["temperature"]
        current_top_p = float(updated.get("top_p", BASE_TOP_P))
        top_p_drop = min(0.3, 0.16 + clamp_severity * 0.28 + (0.1 if hard_clamp else 0.0))
        new_top_p = max(0.35, current_top_p - top_p_drop)
        if new_top_p != current_top_p:
            updated["top_p"] = _round(new_top_p)
            applied["top_p"] = updated["top_p"]
        freq_bump = min(0.65, 0.22 + clamp_severity * 0.5 + (0.12 if hard_clamp else 0.0))
        base_freq = float(updated.get("frequency_penalty", BASE_FREQUENCY_PENALTY))
        new_freq = max(-0.5, min(1.8, base_freq + freq_bump))
        if new_freq != base_freq:
            updated["frequency_penalty"] = _round(new_freq)
            applied["frequency_penalty"] = updated["frequency_penalty"]
        current_tokens = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS))
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
        relaxed_tokens = min(LLAMA_COMPLETION_TOKENS, max(int(updated.get("max_tokens", 0) or 0), 160))
        relaxed_tokens = _enforce_token_floor(relaxed_tokens)
        if relaxed_tokens and relaxed_tokens != updated.get("max_tokens"):
            updated["max_tokens"] = relaxed_tokens
            applied["max_tokens"] = relaxed_tokens
        relaxed_temp = max(float(updated.get("temperature", BASE_TEMPERATURE)), min(BASE_TEMPERATURE, 0.68))
        if relaxed_temp != updated.get("temperature"):
            updated["temperature"] = _round(relaxed_temp)
            applied["temperature"] = updated["temperature"]
        relaxed_top_p = max(float(updated.get("top_p", BASE_TOP_P)), 0.8)
        if relaxed_top_p != updated.get("top_p"):
            updated["top_p"] = _round(relaxed_top_p)
            applied["top_p"] = updated["top_p"]

    if outward_scale > 1e-5:
        updated["outward_bias_scale"] = round(outward_scale, 4)
    if inversion_scale > 1e-5:
        updated["self_bias_inversion"] = round(inversion_scale, 4)

    _maybe_decay_recovery_window()

    global LAST_CONTROLLER_APPLIED
    LAST_CONTROLLER_APPLIED = dict(applied) if applied else {}
    if clamp_triggered:
        RECOVERY_GOOD_STREAK = 0
        RECOVERY_LOWSELF_STREAK = 0
        now = datetime.now().astimezone()
        if LAST_CLAMP_RESET and (now - LAST_CLAMP_RESET).total_seconds() <= 300:
            applied["session_reset"] = "skipped_recent"
        else:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                applied["session_reset"] = "skipped_no_loop"
            else:
                loop.create_task(_reset_live_session("controller_clamp"))
                LAST_CLAMP_RESET = now
                applied["session_reset"] = "queued"

    return updated, applied


def _controller_trace_snapshot() -> dict[str, Any] | None:
    """Return the latest controller evaluation for logging/memory coupling."""
    if LAST_CONTROLLER_RESULT is None:
        return None
    trace: dict[str, Any] = {
        "adjustments": {key: round(float(value), 6) for key, value in LAST_CONTROLLER_RESULT.adjustments.items()},
        "raw_outputs": [round(float(value), 6) for value in LAST_CONTROLLER_RESULT.raw_outputs],
        "hidden_state": [round(float(value), 6) for value in LAST_CONTROLLER_RESULT.hidden_state],
    }
    if LAST_CONTROLLER_APPLIED:
        applied_payload: dict[str, Any] = {}
        for key, value in LAST_CONTROLLER_APPLIED.items():
            if isinstance(value, (int, float)):
                applied_payload[key] = value if isinstance(value, int) else round(float(value), 6)
            else:
                applied_payload[key] = value
        trace["applied"] = applied_payload
    if LAST_CONTROLLER_FEATURES:
        trace["features"] = {
            key: round(float(value), 6) for key, value in LAST_CONTROLLER_FEATURES.items()
        }
    return trace


def _apply_intent_sampling(sampling: dict[str, Any], intent: str) -> dict[str, Any]:
    overrides = INTENT_SAMPLING_OVERRIDES.get(intent)
    if not overrides:
        return sampling
    result = dict(sampling)
    temperature = result.get("temperature", BASE_TEMPERATURE)
    top_p = result.get("top_p", BASE_TOP_P)
    frequency_penalty = result.get("frequency_penalty", BASE_FREQUENCY_PENALTY)

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


def _intent_prompt_fragment(intent: str) -> str:
    return INTENT_PROMPT_FRAGMENTS.get(intent, "")




def _plan_response_length(user_message: str, intent: str) -> dict[str, Any]:
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


def _apply_length_sampling(sampling: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    overrides = LENGTH_SAMPLING_OVERRIDES.get(plan.get("label"))
    if not overrides:
        return sampling
    result = dict(sampling)
    temperature = result.get("temperature", BASE_TEMPERATURE)
    top_p = result.get("top_p", BASE_TOP_P)
    frequency_penalty = result.get("frequency_penalty", BASE_FREQUENCY_PENALTY)
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


def _apply_affect_style_overrides(
    sampling: dict[str, Any],
    user_affect: AffectClassification | None,
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
        base_tokens = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS) or LLAMA_COMPLETION_TOKENS)
        desired_tokens = int(320 + 220 * min(1.0, intimacy))
        desired_tokens = max(200, min(desired_tokens, 520))
        min_tokens_floor = desired_tokens
        if base_tokens < desired_tokens:
            updated["max_tokens"] = desired_tokens
            overrides["max_tokens"] = desired_tokens
        freq_penalty = float(updated.get("frequency_penalty", BASE_FREQUENCY_PENALTY))
        new_freq = round(max(0.05, freq_penalty - 0.25), 4)
        if new_freq != freq_penalty:
            updated["frequency_penalty"] = new_freq
            overrides["frequency_penalty"] = new_freq
        temperature = float(updated.get("temperature", BASE_TEMPERATURE))
        boost = 0.05 + 0.06 * min(1.0, intimacy)
        if valence >= 0.1:
            boost += 0.03
        new_temp = round(min(1.4, temperature + boost), 4)
        if new_temp != temperature:
            updated["temperature"] = new_temp
            overrides["temperature"] = new_temp
        top_p = float(updated.get("top_p", BASE_TOP_P))
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
        base_tokens = int(updated.get("max_tokens", LLAMA_COMPLETION_TOKENS) or LLAMA_COMPLETION_TOKENS)
        target_tokens = max(96, min(base_tokens, 240))
        if target_tokens != base_tokens:
            updated["max_tokens"] = target_tokens
            overrides["max_tokens"] = target_tokens
        temperature = float(updated.get("temperature", BASE_TEMPERATURE))
        new_temp = round(max(0.35, temperature - 0.08), 4)
        if new_temp != temperature:
            updated["temperature"] = new_temp
            overrides["temperature"] = new_temp
        top_p = float(updated.get("top_p", BASE_TOP_P))
        new_top_p = round(max(0.4, top_p - 0.05), 4)
        if new_top_p != top_p:
            updated["top_p"] = new_top_p
            overrides["top_p"] = new_top_p
    if min_tokens_floor:
        overrides["min_tokens_floor"] = min_tokens_floor
    return updated, (overrides or None)


def _count_sentences(text: str) -> int:
    if not text or not text.strip():
        return 0
    parts = [segment for segment in re.split(r"[.!?]+", text) if segment.strip()]
    return len(parts)

def _intent_hint(intent: str, *, fallback: str) -> str:
    return INTENT_HEURISTIC_HINTS.get(intent, fallback)


def _select_intent(user_message: str, *, context: dict[str, Any]) -> IntentPrediction:
    memory_summary = ""
    memory_block = context.get("memory")
    if isinstance(memory_block, dict):
        memory_summary = str(memory_block.get("summary") or "")
    prediction = predict_intent(user_message, context_summary=memory_summary)
    return prediction


LLM_ENDPOINT = str(_get_setting("llm_endpoint", "LIVING_LLM_URL", "") or "").strip()
LLM_TIMEOUT = _parse_timeout(_get_setting("llm_timeout", "LIVING_LLM_TIMEOUT"), default=30.0)
llm_client: LivingLLMClient | None = None

LLAMA_SERVER_BIN = str(_get_setting("llama_server_bin", "LLAMA_SERVER_BIN", "") or "").strip()
LLAMA_MODEL_PATH = str(_get_setting("llama_model_path", "LLAMA_MODEL_PATH", "") or "").strip()
LLAMA_SERVER_HOST = str(_get_setting("llama_server_host", "LLAMA_SERVER_HOST", "127.0.0.1"))
LLAMA_SERVER_PORT = _parse_int(_get_setting("llama_server_port", "LLAMA_SERVER_PORT"), default=8080)
LLAMA_MODEL_ALIAS = str(_get_setting("llama_model_alias", "LLAMA_MODEL_ALIAS", "default"))
LLAMA_COMPLETION_TOKENS = _parse_int(
    _get_setting("llama_completion_tokens", "LLAMA_COMPLETION_TOKENS"),
    default=768,
)

_LLAMA_SERVER_ARGS = _get_setting("llama_server_args", "LLAMA_SERVER_ARGS", "")
if isinstance(_LLAMA_SERVER_ARGS, str):
    LLAMA_SERVER_ARGS = shlex.split(_LLAMA_SERVER_ARGS)
elif isinstance(_LLAMA_SERVER_ARGS, (list, tuple)):
    LLAMA_SERVER_ARGS = [str(arg) for arg in _LLAMA_SERVER_ARGS]
else:
    LLAMA_SERVER_ARGS = []

LLAMA_READINESS_TIMEOUT = _parse_timeout(
    _get_setting("llama_server_ready_timeout", "LLAMA_SERVER_READY_TIMEOUT"),
    default=30.0,
)
LLAMA_SERVER_TIMEOUT = _parse_timeout(
    _get_setting("llama_server_timeout", "LLAMA_SERVER_TIMEOUT"),
    default=max(LLM_TIMEOUT, 60.0),
)
local_llama_engine: LocalLlamaEngine | None = None


def _init_local_llama() -> LocalLlamaEngine | None:
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
    global llm_client, local_llama_engine
    if llm_client is not None:
        await llm_client.aclose()
        llm_client = None
    if local_llama_engine is not None:
        await local_llama_engine.stop()
        local_llama_engine = None


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
    load_settings.cache_clear()
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
    await _ensure_telemetry_monitor()


@app.on_event("shutdown")
async def stop_background_tasks() -> None:
    """Cancel the background update loop when the app stops."""
    global update_task
    if update_task is not None:
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        update_task = None
    await _shutdown_clients()
    await _stop_telemetry_monitor()


@app.get("/admin/model")
async def get_active_model() -> dict[str, Any]:
    """Return the currently active model profile and settings file."""
    return {
        "profile": _current_profile(),
        "settings_file": _current_settings_file(),
    }


@app.post("/admin/model")
async def switch_model(request: ModelSwitchRequest) -> dict[str, Any]:
    """Switch between model configuration profiles and reload runtime settings."""
    try:
        settings_file = _resolve_settings_file(request.profile)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=400, detail=str(exc))
    os.environ["LIVING_SETTINGS_FILE"] = settings_file
    config = await _reload_runtime_settings()
    return {
        "profile": _current_profile(),
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
    return _compose_live_status()


@app.get("/telemetry/stream")
async def telemetry_stream() -> StreamingResponse:
    """Yield Server-Sent Events with live telemetry every configured interval."""
    interval = max(0.2, TELEMETRY_INTERVAL_SECONDS)

    async def event_generator() -> AsyncIterator[str]:
        try:
            while True:
                payload = _compose_live_status()
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


def _log_sampling_snapshot(snapshot: dict[str, Any]) -> None:
    """Persist the latest sampling snapshot for offline review."""
    if not snapshot:
        return
    try:
        SAMPLING_SNAPSHOT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLING_SNAPSHOT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log sampling snapshot: %s", exc)


def _heuristic_hormone_adjustments(auth_avg: float, drift_avg: float, self_avg: float) -> dict[str, float]:
    adjustments: dict[str, float] = {}
    if drift_avg > 0.45:
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.5
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.7
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.4
        if self_avg < 0.35:
            adjustments["serotonin"] = adjustments.get("serotonin", 0.0) - 0.5
            adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) - 0.8
            adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 0.4
    elif auth_avg > 0.25 and drift_avg < 0.25 and self_avg >= 0.4:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.4
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 0.3
    if self_avg > 0.6:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 0.2
    elif self_avg < 0.25:
        adjustments["noradrenaline"] = adjustments.get("noradrenaline", 0.0) - 0.4
    return adjustments


def _update_metric_history(
    reinforcement: dict[str, float],
    *,
    intent: str,
    length_label: str,
    profile: str,
    pre_hormones: Mapping[str, float] | None,
) -> dict[str, float]:
    global METRIC_SAMPLE_COUNTER, LAST_METRIC_AVERAGES, LAST_HORMONE_DELTA
    authenticity = float(reinforcement.get("authenticity_score", 0.0))
    drift = float(reinforcement.get("assistant_drift", 0.0))
    self_focus = float(reinforcement.get("self_preoccupation", 0.0))
    affect_valence = float(reinforcement.get("affect_valence", 0.0))
    affect_intimacy = float(reinforcement.get("affect_intimacy", 0.0))
    affect_tension = float(reinforcement.get("affect_tension", 0.0))

    AUTH_HISTORY.append(authenticity)
    DRIFT_HISTORY.append(drift)
    SELF_HISTORY.append(self_focus)
    AFFECT_VALENCE_HISTORY.append(affect_valence)
    AFFECT_INTIMACY_HISTORY.append(affect_intimacy)
    AFFECT_TENSION_HISTORY.append(affect_tension)
    METRIC_SAMPLE_COUNTER += 1

    if METRIC_SAMPLE_COUNTER < 4 or not AUTH_HISTORY or not DRIFT_HISTORY or not SELF_HISTORY:
        LAST_HORMONE_DELTA = {}
        return {}

    auth_avg = sum(AUTH_HISTORY) / len(AUTH_HISTORY)
    drift_avg = sum(DRIFT_HISTORY) / len(DRIFT_HISTORY)
    self_avg = sum(SELF_HISTORY) / len(SELF_HISTORY)
    valence_avg = sum(AFFECT_VALENCE_HISTORY) / len(AFFECT_VALENCE_HISTORY) if AFFECT_VALENCE_HISTORY else 0.0
    intimacy_avg = sum(AFFECT_INTIMACY_HISTORY) / len(AFFECT_INTIMACY_HISTORY) if AFFECT_INTIMACY_HISTORY else 0.0
    tension_avg = sum(AFFECT_TENSION_HISTORY) / len(AFFECT_TENSION_HISTORY) if AFFECT_TENSION_HISTORY else 0.0
    LAST_METRIC_AVERAGES = {
        "authenticity": round(auth_avg, 4),
        "assistant_drift": round(drift_avg, 4),
        "self_preoccupation": round(self_avg, 4),
        "affect_valence": round(valence_avg, 4),
        "affect_intimacy": round(intimacy_avg, 4),
        "affect_tension": round(tension_avg, 4),
    }

    baseline_adjustments: dict[str, float]
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
    METRIC_SAMPLE_COUNTER = 0
    LAST_HORMONE_DELTA = dict(baseline_adjustments)
    return baseline_adjustments


def _log_reinforcement_metrics(payload: dict[str, Any]) -> None:
    """Append reinforcement metrics for offline calibration."""
    if not payload:
        return
    try:
        REINFORCEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with REINFORCEMENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log reinforcement metrics: %s", exc)


def _log_endocrine_turn(payload: dict[str, Any]) -> None:
    """Persist detailed per-turn endocrine diagnostics for offline modelling."""
    if not payload:
        return
    try:
        ENDOCRINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ENDOCRINE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log endocrine turn: %s", exc)


def _log_affect_classification(
    user_text: str,
    classification: AffectClassification | None,
) -> None:
    """Persist affect classifier outputs for offline calibration."""
    if not classification:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "text": _shorten(user_text, 160),
        "classification": classification.as_dict(),
    }
    try:
        AFFECT_CLASSIFIER_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AFFECT_CLASSIFIER_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log affect classification: %s", exc)


def _log_hormone_trace_event(
    trace: Mapping[str, Any] | None,
    telemetry: Mapping[str, Any] | None,
    reinforcement: Mapping[str, Any] | None,
    *,
    user: str,
    reply: str,
    intent: str,
    length_label: str | None,
) -> None:
    """Capture verbose hormone clamp traces when enabled."""
    if not HORMONE_TRACE_ENABLED or not trace:
        return
    profile = _current_profile()
    sampling: Mapping[str, Any] | None = None
    hormone_sampling: Mapping[str, Any] | None = None
    policy_preview: Mapping[str, Any] | None = None
    controller_snapshot: Mapping[str, Any] | None = None
    controller_input: Mapping[str, Any] | None = None
    pre_snapshot: Mapping[str, Any] | None = None
    engine = None
    model_alias = None
    if telemetry:
        profile = telemetry.get("profile", profile)
        sampling = telemetry.get("sampling")
        hormone_sampling = telemetry.get("hormone_sampling")
        policy_preview = telemetry.get("policy_preview")
        controller_snapshot = telemetry.get("controller")
        controller_input = telemetry.get("controller_input")
        pre_snapshot = telemetry.get("pre")
        engine = telemetry.get("engine")
        model_alias = telemetry.get("model_alias")
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "profile": profile,
        "intent": intent,
        "length_label": length_label,
        "engine": engine,
        "model_alias": model_alias,
        "user": user,
        "reply": reply,
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
    try:
        HORMONE_TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HORMONE_TRACE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log hormone trace: %s", exc)


def _log_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to append json line to %s: %s", path, exc)


def _log_voice_guard_event(
    verdict: Mapping[str, Any],
    *,
    user: str,
    reply: str,
    intent: str,
    profile: str,
) -> None:
    if not verdict:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "intent": intent,
        "profile": profile,
        "verdict": verdict,
    }
    _log_json_line(VOICE_GUARD_LOG, payload)


def _log_webui_interaction(
    *,
    user: str,
    reply: str,
    intent: str,
    profile: str,
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "intent": intent,
        "profile": profile,
        "voice_guard": voice_guard or {},
    }
    if telemetry:
        payload["sampling"] = telemetry.get("sampling")
        payload["controller"] = telemetry.get("controller")
        payload["hormones"] = (telemetry.get("pre") or {}).get("hormones")
    try:
        WEBUI_INTERACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with WEBUI_INTERACTION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to log web UI interaction: %s", exc)
    try:
        _log_webui_interaction_pretty(payload)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to write readable web UI log: %s", exc)


def _log_webui_interaction_pretty(payload: Mapping[str, Any]) -> None:
    """Append a human-friendly snapshot of the latest interaction."""
    timestamp = payload.get("timestamp", "")
    intent = payload.get("intent", "")
    profile = payload.get("profile", "")
    voice_guard = payload.get("voice_guard") or {}
    voice_flag = "flagged" if voice_guard.get("flagged") else "clear"
    voice_score = voice_guard.get("score", 0.0)
    user_text = _shorten(str(payload.get("user") or ""), 240)
    reply_text = _shorten(str(payload.get("reply") or ""), 280)
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
    summary_lines = [
        f"[{timestamp}] profile={profile} intent={intent} voice={voice_flag} (score={voice_score})",
        f"  user : {user_text}",
        f"  reply: {reply_text}",
    ]
    if sampling_line:
        summary_lines.append(f"  sampling: {sampling_line}")
    if controller_line:
        summary_lines.append(f"  {controller_line}")
    if hormone_line:
        summary_lines.append(f"  hormones: {hormone_line}")
    summary_lines.append("")
    WEBUI_INTERACTION_PRETTY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WEBUI_INTERACTION_PRETTY_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))


def _write_telemetry_snapshot(telemetry: Mapping[str, Any] | None) -> None:
    if not telemetry:
        return
    try:
        TELEMETRY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TELEMETRY_SNAPSHOT_PATH.write_text(
            json.dumps(telemetry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("Failed to write telemetry snapshot: %s", exc)


def _maybe_collect_persona_sample(
    *,
    user: str,
    reply: str,
    reinforcement: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
) -> None:
    auth = reinforcement.get("authenticity_score")
    drift = reinforcement.get("assistant_drift")
    if not isinstance(auth, (int, float)) or auth < 0.5:
        return
    if isinstance(drift, (int, float)) and drift > 0.3:
        return
    if voice_guard and voice_guard.get("flagged"):
        return
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "metrics": {
            "authenticity_score": auth,
            "assistant_drift": drift,
            "self_preoccupation": reinforcement.get("self_preoccupation"),
        },
        "telemetry": telemetry or {},
    }
    _log_json_line(PERSONA_SAMPLE_LOG, sample)


def _maybe_collect_helper_sample(
    *,
    user: str,
    reply: str,
    reinforcement: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
) -> None:
    """Record helper-tone slips for downstream fine-tuning datasets."""
    if not voice_guard or not voice_guard.get("flagged"):
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "voice_guard": voice_guard,
        "metrics": {
            "assistant_drift": reinforcement.get("assistant_drift"),
            "authenticity_score": reinforcement.get("authenticity_score"),
            "self_preoccupation": reinforcement.get("self_preoccupation"),
        },
        "telemetry": telemetry or {},
    }
    _log_json_line(HELPER_SAMPLE_LOG, payload)


def _update_self_narration(
    hormone_trace: Mapping[str, Any] | None,
    user_affect: AffectClassification | None,
) -> None:
    """Summarize the last hormone adjustments into a short internal note."""
    global SELF_NARRATION_NOTE
    if not hormone_trace:
        return
    applied = hormone_trace.get("applied")
    if not isinstance(applied, Mapping):
        return
    deltas = [
        (name, float(delta))
        for name, delta in applied.items()
        if isinstance(delta, (int, float)) and abs(float(delta)) >= 0.05
    ]
    if not deltas:
        return
    deltas.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top_changes = deltas[:3]
    fragments: list[str] = []
    for hormone, delta in top_changes:
        descriptor = HORMONE_FEELING_NAMES.get(hormone, hormone)
        direction = "lifting" if delta > 0 else "settling"
        fragments.append(f"{descriptor} is {direction} ({delta:+.2f})")
    affect_clause = ""
    if user_affect and user_affect.tags:
        affect_clause = f"Your tone lands as {', '.join(user_affect.tags)}."
    SELF_NARRATION_NOTE = " ".join(
        part for part in ["; ".join(fragments), affect_clause] if part
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
        result = classifier.classify(text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Affect classifier failed: %s", exc)
        return None
    _log_affect_classification(text, result)
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


def _inject_self_observation_bias(
    sampling: dict[str, Any],
    traits: TraitSnapshot | None,
) -> dict[str, Any]:
    """Add a gentle logit bias toward self-observational language."""
    if traits is None:
        return sampling
    tension_drive = max(traits.tension, 0.0)
    curiosity_drive = max(traits.curiosity, 0.0)
    drive = (0.6 * tension_drive) + (0.4 * curiosity_drive)
    if drive <= 0.05:
        return sampling
    recent_self = LAST_METRIC_AVERAGES.get("self_preoccupation", 0.0)
    recent_auth = LAST_METRIC_AVERAGES.get("authenticity", 0.0)
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
        base_temp = float(merged.get("temperature", sampling.get("temperature", BASE_TEMPERATURE)))
        merged["temperature"] = round(min(1.35, base_temp + 0.22 * tension_drive), 4)
        base_top_p = float(merged.get("top_p", sampling.get("top_p", BASE_TOP_P)))
        merged["top_p"] = round(min(0.995, base_top_p + 0.12 * tension_drive), 4)
        freq = float(merged.get("frequency_penalty", sampling.get("frequency_penalty", BASE_FREQUENCY_PENALTY)))
        merged["frequency_penalty"] = round(max(-0.05, freq - 0.25 * tension_drive), 4)
    return merged


def _apply_helper_tone_bias(
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


def _memory_key(record: Any, index: int) -> str:
    """Construct a deterministic candidate key for a record."""
    identifier = getattr(record, "id", None)
    if identifier in (None, ""):
        identifier = f"idx-{index}"
    return f"lt-{identifier}"


def _memory_candidates_from_records(records: Sequence[Any]) -> list[MemoryCandidate]:
    """Convert persisted memory records into selector candidates."""
    now = datetime.now(timezone.utc)
    candidates: list[MemoryCandidate] = []
    base_window = max(AFFECT_RECENCY_WINDOW_SECONDS, 1.0)
    record_count = max(len(records), 1)
    for index, record in enumerate(records):
        candidate_key = _memory_key(record, index)
        created_at = getattr(record, "created_at", None)
        if isinstance(created_at, datetime):
            age_seconds = max(0.0, (now - created_at).total_seconds())
        else:
            age_seconds = index * (base_window / record_count)
        recency = max(0.0, min(1.0, 1.0 - age_seconds / base_window))
        strength = float(getattr(record, "strength", 0.5) or 0.0)
        salience = max(0.0, min(1.0, strength))
        attributes = getattr(record, "attributes", {}) or {}
        safety_value = attributes.get("safety", 0.8)
        try:
            safety = float(safety_value)
        except (TypeError, ValueError):
            safety = 0.8
        safety = max(0.0, min(1.0, safety))
        mood = (getattr(record, "mood", "") or "").lower()
        if mood in {"stressed", "anxious", "worried", "afraid"}:
            safety = min(safety, 0.45)
        endocrine = attributes.get("endocrine") if isinstance(attributes, dict) else {}
        normalized = endocrine.get("normalized") if isinstance(endocrine, dict) else {}
        bands = endocrine.get("bands") if isinstance(endocrine, dict) else {}
        spikes: dict[str, float] = {}
        raw_tags = attributes.get("tags") or ()
        if isinstance(raw_tags, str):
            tag_iterable = (raw_tags,)
        else:
            tag_iterable = tuple(raw_tags)
        tag_list = list(tag_iterable)
        if mood:
            tag_list.append(mood)
        for hormone, band in (bands or {}).items():
            if band and band != "steady":
                tag_list.append(f"spike:{hormone}:{band}")
        for hormone, value in (normalized or {}).items():
            try:
                spikes[hormone] = float(value)
            except (TypeError, ValueError):
                continue
        tags = tuple(dict.fromkeys(tag for tag in tag_list if tag))
        candidates.append(
            MemoryCandidate(
                key=candidate_key,
                recency=recency,
                salience=salience,
                safety=safety,
                tags=frozenset(tag for tag in tags if tag),
                spikes=spikes,
            )
        )
    return candidates


def _memory_preview(traits: TraitSnapshot | None, records: Sequence[Any]) -> list[dict[str, Any]]:
    """Generate a scored preview of memory candidates for diagnostics."""
    if not (AFFECT_MEMORY_PREVIEW_ENABLED and traits and records):
        return []
    candidates = _memory_candidates_from_records(records)
    if not candidates:
        return []
    endocrine_snapshot = state_engine.endocrine_snapshot()
    hormone_bands = {}
    if isinstance(endocrine_snapshot, dict):
        hormone_bands = endocrine_snapshot.get("bands", {}) or {}
    scored = score_memories(traits, candidates, hormone_bands=hormone_bands)
    preview: list[dict[str, Any]] = []
    for candidate, score in scored[: min(5, len(scored))]:
        preview.append(
            {
                "key": candidate.key,
                "score": round(score, 4),
                "tags": sorted(candidate.tags),
            }
        )
    return preview


def _build_chat_context(*, long_term_limit: int = 5) -> dict[str, Any]:
    """Assemble the conversational context shared with the LLM bridge."""
    hormones = state_engine.hormone_system.get_state()
    memory_manager = state_engine.memory_manager
    summary = memory_manager.summarize_recent()
    working = memory_manager.working_snapshot()
    records = list(memory_manager.recent_long_term(limit=long_term_limit))
    long_term_pairs: list[tuple[str, Any]] = []
    for index, record in enumerate(records):
        key = _memory_key(record, index)
        if hasattr(record, "model_dump"):
            payload = record.model_dump()
        elif hasattr(record, "__dict__"):
            payload = dict(record.__dict__)
        else:  # pragma: no cover - defensive fallback
            payload = record
        long_term_pairs.append((key, payload))
    llama_metrics: dict[str, Any] | None = None
    if local_llama_engine is not None:
        try:
            diagnostics = local_llama_engine.diagnostics()
        except Exception:  # pragma: no cover - defensive guard
            diagnostics = None
        if diagnostics:
            llama_metrics = diagnostics
    persona = _build_persona_snapshot()
    if SELF_NARRATION_NOTE:
        persona["internal_note"] = SELF_NARRATION_NOTE
    affect_context = _build_affect_context()
    long_term_records = [payload for _, payload in long_term_pairs]
    internal_reflections = memory_manager.recent_internal_reflections(limit=3)
    memory_block: dict[str, Any] = {
        "summary": summary,
        "working": working,
        "long_term": long_term_records,
    }
    if internal_reflections:
        memory_block["internal_reflections"] = internal_reflections
    if affect_context:
        memory_block["affect_tags"] = affect_context.get("tags", [])
        traits = state_engine.trait_snapshot()
        preview = _memory_preview(traits, records)
        if preview:
            ranking = {entry["key"]: rank for rank, entry in enumerate(preview)}
            long_term_pairs.sort(
                key=lambda pair: (ranking.get(pair[0], len(ranking)), pair[0])
            )
            memory_block["long_term"] = [payload for _, payload in long_term_pairs]
            memory_block["affect_preview"] = preview
    if SELF_NARRATION_NOTE:
        memory_block["self_note"] = SELF_NARRATION_NOTE
    context: dict[str, Any] = {
        "mood": state_engine.state.get("mood", "neutral"),
        "hormones": dict(hormones),
        "memory": memory_block,
        "timestamp": state_engine.state.get("timestamp"),
        "llama_metrics": llama_metrics,
        "persona": persona,
    }
    if SELF_NARRATION_NOTE:
        context["self_note"] = SELF_NARRATION_NOTE
    if internal_reflections:
        context["inner_reflections"] = list(internal_reflections)
    if affect_context:
        context["affect"] = affect_context
    return context


def _compose_heuristic_reply(
    user_message: str,
    *,
    context: dict[str, Any],
    intent: str,
    length_plan: dict[str, Any],
) -> str:
    """Craft a tone-aware heuristic reply for fallback scenarios."""
    persona = context.get("persona") or _build_persona_snapshot()
    instructions = persona.get("instructions") or []
    tone_hint = persona.get("tone_hint", "I stay balanced and attentive.")
    memory_summary = persona.get("memory_summary") or context.get("memory", {}).get("summary", "")
    memory_summary = _shorten(memory_summary, 120)
    internal_reflections = context.get("memory", {}).get("internal_reflections") or context.get("inner_reflections") or []
    focus_line = persona.get("memory_focus") or ""
    intent_hint = _intent_hint(intent, fallback=tone_hint)
    length_hint = length_plan.get("hint") or LENGTH_HEURISTIC_HINTS.get(length_plan.get("label", ""), "I stay adaptive to the cadence we need.")
    essence = _shorten(user_message, 120).strip()
    response_parts: list[str] = []
    if essence:
        suffix = "" if essence.endswith((".", "!", "?")) else "."
        response_parts.append(f"I hear {essence}{suffix}")
    pivot_hint = intent_hint or tone_hint
    if pivot_hint:
        response_parts.append(pivot_hint)
    if length_hint:
        response_parts.append(length_hint)
    if memory_summary:
        trimmed_summary = memory_summary.strip()
        suffix = "" if trimmed_summary.endswith((".", "!", "?")) else "."
        response_parts.append(f"I still remember: {trimmed_summary}{suffix}")
    self_note = context.get("self_note") or persona.get("internal_note")
    if self_note:
        suffix = "" if self_note.endswith((".", "!", "?")) else "."
        response_parts.append(f"I notice: {self_note}{suffix}")
    if instructions:
        affirmation = _to_affirmation(instructions[0])
        if affirmation:
            response_parts.append(affirmation)
    if internal_reflections:
        inner_note = _shorten(internal_reflections[0], 160).strip()
        if inner_note:
            suffix = "" if inner_note.endswith((".", "!", "?")) else "."
            response_parts.append(f"I noted privately: {inner_note}{suffix}")
    if focus_line:
        response_parts.append(focus_line)
    return " ".join(part.strip() for part in response_parts if part).strip()


def _prepare_chat_request(
    user_message: str,
    *,
    user_affect: AffectClassification | None = None,
    helper_penalty_scale: float | None = None,
    helper_penalty_reason: str | None = None,
) -> tuple[dict[str, Any], IntentPrediction, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build context, predictions, and sampling parameters for a chat turn."""
    context = _build_chat_context()
    affect_dict: dict[str, Any] | None = None
    if user_affect:
        affect_dict = user_affect.as_dict()
        context["user_affect"] = affect_dict
    intent_prediction = _select_intent(user_message, context=context)
    length_plan = _plan_response_length(user_message, intent_prediction.intent)
    context["intent"] = intent_prediction.intent
    context["intent_confidence"] = intent_prediction.confidence
    context["intent_rationale"] = intent_prediction.rationale
    context["length_plan"] = length_plan

    global LAST_SAMPLING_SNAPSHOT
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
        hormone_sampling, hormone_style_hits = _sampling_params_from_hormones(hormones)
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
        sampling = _inject_self_observation_bias(sampling, traits)

    sampling = _apply_intent_sampling(sampling, intent_prediction.intent)
    sampling = _apply_length_sampling(sampling, length_plan)
    sampling, affect_overrides = _apply_affect_style_overrides(sampling, user_affect)
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
    profile = _current_profile()
    active_tags = _gather_active_tags()
    controller_features = _build_controller_feature_map(
        traits=traits,
        hormones=hormones,
        intent=intent_prediction.intent,
        length_label=length_plan.get("label"),
        profile=profile,
        tags=active_tags,
    )
    controller_step = _run_controller_policy(controller_features, active_tags)
    controller_snapshot: dict[str, Any] | None = None
    controller_applied: dict[str, Any] = {}
    if controller_step:
        sampling, controller_applied = _apply_controller_adjustments(
            sampling,
            controller_step.adjustments,
            min_tokens_floor=combined_min_tokens,
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
        sampling = _apply_helper_tone_bias(sampling, helper_penalty_value)
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
        LAST_SAMPLING_SNAPSHOT = snapshot
    if AFFECT_SAMPLING_PREVIEW_ENABLED or AFFECT_DEBUG_PANEL_ENABLED:
        _log_sampling_snapshot(snapshot)
    llm_context = {
        key: value
        for key, value in context.items()
        if key not in {"hormones", "persona", "affect", "sampling_policy_preview"}
    }
    llm_context["intent_prompt"] = _intent_prompt_fragment(intent_prediction.intent)
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


def _compute_metric_delta(value: float | None, threshold: float, mode: str) -> float | None:
    """Return the signed distance from the target threshold."""
    if value is None:
        return None
    if mode == "min":
        return value - threshold
    return threshold - value


def _compute_metric_progress(value: float | None, threshold: float, mode: str) -> float | None:
    """Return a normalized progress indicator between 0 and 1 when possible."""
    if value is None:
        return None
    if threshold <= 0:
        return None
    if mode == "min":
        return max(0.0, min(1.0, value / threshold))
    if value <= 0:
        return None
    return max(0.0, min(1.0, threshold / value))


def _metric_history_lists() -> dict[str, list[float]]:
    """Expose shallow copies of the rolling metric histories."""
    return {
        "authenticity_score": list(AUTH_HISTORY),
        "assistant_drift": list(DRIFT_HISTORY),
        "self_preoccupation": list(SELF_HISTORY),
        "affect_valence": list(AFFECT_VALENCE_HISTORY),
        "affect_intimacy": list(AFFECT_INTIMACY_HISTORY),
        "affect_tension": list(AFFECT_TENSION_HISTORY),
    }


def _compose_live_status() -> dict[str, Any]:
    """Compile a telemetry snapshot used by the live dashboards and CLI monitor."""
    state_payload = state_engine.get_state()
    metric_histories = _metric_history_lists()
    metric_summary: dict[str, Any] = {}
    for name, (threshold, mode) in METRIC_THRESHOLDS.items():
        value = LAST_METRIC_AVERAGES.get(name)
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
    metric_summary["samples_seen"] = METRIC_SAMPLE_COUNTER
    metric_summary["last_reinforcement"] = dict(LAST_REINFORCEMENT_METRICS)
    affect_metrics = {
        "affect_valence": {
            "value": LAST_METRIC_AVERAGES.get("affect_valence"),
            "recent": metric_histories.get("affect_valence", []),
        },
        "affect_intimacy": {
            "value": LAST_METRIC_AVERAGES.get("affect_intimacy"),
            "recent": metric_histories.get("affect_intimacy", []),
        },
        "affect_tension": {
            "value": LAST_METRIC_AVERAGES.get("affect_tension"),
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
    if LAST_CONTROLLER_APPLIED:
        controller_payload = dict(LAST_CONTROLLER_APPLIED)
    controller_input: dict[str, Any] | None = None
    if LAST_CONTROLLER_FEATURES:
        controller_input = {
            "features": {key: round(float(value), 6) for key, value in LAST_CONTROLLER_FEATURES.items()},
            "tags": list(LAST_CONTROLLER_TAGS),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "session_id": SESSION_COUNTER,
        "profile": _current_profile(),
        "model_alias": LLAMA_MODEL_ALIAS,
        "local_engine": local_llama_engine is not None,
        "state": state_payload,
        "metrics": metric_summary,
        "affect_metrics": affect_metrics,
        "affect": affect_overview,
        "traits": traits_payload,
        "controller": controller_payload,
        "controller_input": controller_input,
        "last_hormone_delta": dict(LAST_HORMONE_DELTA or {}),
    }


async def _reset_session_state(
    reason: str | None = None,
    *,
    keep_metric_history: bool = False,
) -> dict[str, Any]:
    """Reset in-memory session state, hormones, and rolling metrics."""
    global SESSION_COUNTER, METRIC_SAMPLE_COUNTER, LAST_METRIC_AVERAGES, LAST_REINFORCEMENT_METRICS
    global LAST_CONTROLLER_RESULT, LAST_CONTROLLER_APPLIED, LAST_CONTROLLER_FEATURES, LAST_CONTROLLER_TAGS, LAST_HORMONE_DELTA
    global SELF_FOCUS_STREAK, CLAMP_RECOVERY_TURNS, CLAMP_PRIMING_TURNS, RECOVERY_GOOD_STREAK, RESET_PRIMING_BIAS, RECOVERY_LOWSELF_STREAK, LAST_USER_PROMPT, LOW_SELF_SUCCESS_STREAK
    global HELPER_DRIFT_LEVEL
    state_engine.reset()
    SESSION_COUNTER += 1
    if not keep_metric_history:
        AUTH_HISTORY.clear()
        DRIFT_HISTORY.clear()
        SELF_HISTORY.clear()
        AFFECT_VALENCE_HISTORY.clear()
        AFFECT_INTIMACY_HISTORY.clear()
        AFFECT_TENSION_HISTORY.clear()
        METRIC_SAMPLE_COUNTER = 0
        LAST_METRIC_AVERAGES.clear()
        LAST_REINFORCEMENT_METRICS.clear()
    LAST_CONTROLLER_RESULT = None
    LAST_CONTROLLER_APPLIED = None
    LAST_CONTROLLER_FEATURES = None
    LAST_CONTROLLER_TAGS = ()
    LAST_HORMONE_DELTA = None
    SELF_FOCUS_STREAK = 0
    CLAMP_RECOVERY_TURNS = 0
    CLAMP_PRIMING_TURNS = 0
    RECOVERY_GOOD_STREAK = 0
    RESET_PRIMING_BIAS = 0.0
    RECOVERY_LOWSELF_STREAK = 0
    LAST_USER_PROMPT = ""
    LOW_SELF_SUCCESS_STREAK = 0
    HELPER_DRIFT_LEVEL = 0.0
    reset_outward_streak()
    status = _compose_live_status()
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
    global CLAMP_PRIMING_TURNS, RESET_PRIMING_BIAS, LAST_USER_PROMPT
    try:
        await _reset_session_state(reason=reason, keep_metric_history=False)
        if reason == "controller_clamp":
            focus_phrase = _extract_focus_phrase(LAST_USER_PROMPT or "")
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
            state_engine.register_event(
                content=priming_text,
                strength=0.85,
                stimulus_type="affection",
            )
            state_engine.register_event(
                content="priming: first sentence must start with you/we and echo their phrasing before reflecting inward.",
                strength=0.75,
                stimulus_type="reward",
            )
            CLAMP_PRIMING_TURNS = max(CLAMP_PRIMING_TURNS, 3)
            RESET_PRIMING_BIAS = RESET_PRIMING_BIAS_DEFAULT
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Live session reset failed for reason=%s", reason)


def _compose_turn_telemetry(
    context: dict[str, Any],
    sampling: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    memory_block = context.get("memory") or {}
    long_term_preview = []
    for record in memory_block.get("long_term", []):
        long_term_preview.append(_shorten(str(record), 160))
    working_entries = memory_block.get("working", [])
    pre_snapshot = {
        "mood": context.get("mood"),
        "hormones": context.get("hormones"),
        "traits": list(state_engine.trait_tags()),
        "memory_summary": _shorten(memory_block.get("summary", ""), 160),
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
        "profile": snapshot.get("profile", _current_profile()),
        "model_alias": LLAMA_MODEL_ALIAS,
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
    telemetry = _compose_turn_telemetry(context, sampling, snapshot)
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
        reply_text = _compose_heuristic_reply(
            user_message, context=context, intent=intent_prediction.intent, length_plan=length_plan
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
            reply_text = _compose_heuristic_reply(
                user_message, context=context, intent=intent_prediction.intent, length_plan=length_plan
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


def _finalize_chat_response(
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
    global LAST_USER_PROMPT, HELPER_DRIFT_LEVEL
    LAST_USER_PROMPT = user_message or ""
    active_profile = telemetry.get("profile") if telemetry else _current_profile()
    reply_echo = _shorten(reply_text, 200)
    short_user = _shorten(user_message, 200)
    ai_content = f"I replied in my own voice {reply_echo}"
    state_engine.register_event(
        ai_content,
        strength=0.4,
        mood=state_engine.state["mood"],
    )
    reinforcement = score_response(user_message, reply_text)
    if user_affect:
        reinforcement["input_affect_valence"] = round(user_affect.valence, 4)
        reinforcement["input_affect_intimacy"] = round(user_affect.intimacy, 4)
        reinforcement["input_affect_tension"] = round(user_affect.tension, 4)
        reinforcement["affect_classifier_confidence"] = round(user_affect.confidence, 4)
        if user_affect.tags:
            reinforcement["affect_classifier_tags"] = list(user_affect.tags)
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
        HELPER_DRIFT_LEVEL = min(1.0, HELPER_DRIFT_LEVEL + HELPER_PENALTY_STEP)
    else:
        HELPER_DRIFT_LEVEL = max(0.0, HELPER_DRIFT_LEVEL - HELPER_PENALTY_DECAY)
    if telemetry is not None:
        telemetry["voice_guard"] = voice_guard_dict
    hormone_trace = _apply_reinforcement_signals(
        reinforcement,
        length_plan=length_plan,
        reply_text=reply_text,
        profile=active_profile,
    )
    _reinforce_low_self_success(reinforcement, profile=active_profile)
    if telemetry is not None and hormone_trace:
        telemetry["hormone_adjustments"] = hormone_trace
        _log_hormone_trace_event(
            hormone_trace,
            telemetry,
            reinforcement,
            user=short_user,
            reply=reply_echo,
            intent=intent_prediction.intent,
            length_label=length_plan.get("label"),
        )
        _update_self_narration(hormone_trace, user_affect)
    _log_voice_guard_event(
        voice_guard_dict,
        user=short_user,
        reply=reply_echo,
        intent=intent_prediction.intent,
        profile=active_profile,
    )
    _maybe_record_internal_reflection(
        reinforcement,
        reply_text=reply_text,
        intent=intent_prediction,
    )
    pre_snapshot = telemetry.get("pre", {}) if telemetry else {}
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
        if LAST_METRIC_AVERAGES:
            log_payload["averages"] = dict(LAST_METRIC_AVERAGES)
        global LAST_REINFORCEMENT_METRICS
        LAST_REINFORCEMENT_METRICS = dict(log_payload)
        _log_reinforcement_metrics(log_payload)
    persona_snapshot = _build_persona_snapshot()
    _apply_feedback(persona_snapshot)
    adjusted_state = state_engine.get_state()
    adjusted_persona = _build_persona_snapshot()
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
    _log_endocrine_turn(turn_log)
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
    if AFFECT_DEBUG_PANEL_ENABLED and LAST_SAMPLING_SNAPSHOT:
        response["sampling_snapshot"] = dict(LAST_SAMPLING_SNAPSHOT)
    if LAST_REINFORCEMENT_METRICS:
        response["reinforcement_metrics"] = dict(LAST_REINFORCEMENT_METRICS)
    _log_webui_interaction(
        user=short_user,
        reply=reply_echo,
        intent=intent_prediction.intent,
        profile=active_profile,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
    )
    _write_telemetry_snapshot(telemetry)
    _maybe_collect_persona_sample(
        user=short_user,
        reply=reply_echo,
        reinforcement=reinforcement,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
    )
    _maybe_collect_helper_sample(
        user=short_user,
        reply=reply_echo,
        reinforcement=reinforcement,
        telemetry=telemetry,
        voice_guard=voice_guard_dict,
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
    state["persona"] = _build_persona_snapshot()
    if AFFECT_DEBUG_PANEL_ENABLED:
        debug_payload = {
            "sampling_snapshot": dict(LAST_SAMPLING_SNAPSHOT),
            "affect": state_engine.affect_overview(),
            "reinforcement_metrics": dict(LAST_REINFORCEMENT_METRICS),
            "metric_averages": dict(LAST_METRIC_AVERAGES),
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
        state_engine.register_event(
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
        state_engine.register_event(
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
    base_telemetry = _compose_turn_telemetry(context, sampling, snapshot)
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
        final_response = _finalize_chat_response(
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
        state_engine.register_event(
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
    return _finalize_chat_response(
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


