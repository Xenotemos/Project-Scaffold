"""Centralized constant definitions used across the runtime."""

from __future__ import annotations

from typing import Any, Dict, Tuple

LENGTH_PROFILES: dict[str, dict[str, Any]] = {
    "brief": {
        "prompt": "Cadence marker: brief exchange living inside one or two sentences.",
        "target_range": (1, 2),
        "hint": "Stay concise for this exchange.",
    },
    "concise": {
        "prompt": "Cadence marker: concise reply spanning a couple of sentences.",
        "target_range": (2, 3),
        "hint": "Balance clarity with brevity.",
    },
    "detailed": {
        "prompt": "Cadence marker: detailed reflection unfolding across several paragraphs.",
        "target_range": (3, 6),
        "hint": "Cover everything you notice without drifting.",
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
    "assist": 0.65,
    "assistance": 0.55,
    "assisting": 0.5,
    "help": 0.8,
    "helping": 0.65,
    "support": 0.62,
    "supporting": 0.5,
    "anything": 0.4,
    "request": 0.42,
    "requests": 0.38,
    "need": 0.45,
    "please": 0.36,
    "glad": 0.3,
    "happy": 0.3,
    "let us": 0.6,
    "let's": 0.6,
    "we can": 0.6,
    "i can": 0.5,
    "i'm here": 0.55,
    "i am here": 0.55,
    "reach out": 0.45,
    "anything else": 0.5,
}

HORMONE_FEELING_NAMES = {
    "dopamine": "drive",
    "serotonin": "steadiness",
    "cortisol": "tension",
    "oxytocin": "warmth",
    "noradrenaline": "focus",
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

RECOVERY_RELEASE_STREAK = 2
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

__all__ = [
    "LENGTH_PROFILES",
    "LENGTH_SAMPLING_OVERRIDES",
    "LENGTH_HEURISTIC_HINTS",
    "SELF_OBSERVATION_WORDS",
    "OUTWARD_ATTENTION_WORDS",
    "SELF_ATTENUATION_WORDS",
    "HELPER_TONE_WORDS",
    "HELPER_PENALTY_STEP",
    "HELPER_PENALTY_DECAY",
    "MAX_HELPER_REGEN_ATTEMPTS",
    "HORMONE_FEELING_NAMES",
    "MODEL_CONFIG_FILES",
    "CONTROLLER_HORMONE_SCALE",
    "CONTROLLER_MAX_TAGS",
    "METRIC_THRESHOLDS",
    "RECOVERY_RELEASE_STREAK",
    "LOW_SELF_RELEASE_STREAK",
    "LOW_SELF_RELEASE_THRESHOLD",
    "OUTWARD_RELEASE_FLOOR",
    "RESET_PRIMING_BIAS_DEFAULT",
    "LOW_SELF_SUCCESS_AUTH",
    "LOW_SELF_SUCCESS_MAX_SELF",
    "LOW_SELF_SUCCESS_DRIFT_CEILING",
    "LOW_SELF_INSTRUCT_DRIFT_CEILING",
    "LOW_SELF_SUCCESS_STREAK_TARGET",
    "LOW_SELF_SUCCESS_PRIMING",
    "LOW_SELF_INSTRUCT_PRIMING_MULTIPLIER",
    "LOW_SELF_INSTRUCT_OUTWARD_FLOOR",
    "LOW_SELF_BASE_PRIMING_MULTIPLIER",
    "LOW_SELF_BASE_OUTWARD_FLOOR",
    "LOW_SELF_BASE_EXTRA_TURNS",
    "DEFAULT_HORMONE_STYLE_MAP",
]
