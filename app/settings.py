"""Runtime settings loader and related helpers."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Any, Sequence

from utils.settings import load_settings


def _parse_float(value: Any, default: float) -> float:
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


def _get_setting(settings: dict[str, Any], key: str, env_var: str, default: Any = None) -> Any:
    value = settings.get(key)
    if value not in (None, ""):
        return value
    env_value = os.getenv(env_var)
    if env_value not in (None, ""):
        return env_value
    return default


@dataclass(frozen=True)
class RuntimeSettings:
    raw: dict[str, Any]
    base_temperature: float
    base_top_p: float
    base_frequency_penalty: float
    llm_endpoint: str
    llm_timeout: float
    llama_server_bin: str
    llama_model_path: str
    llama_server_host: str
    llama_server_port: int
    llama_model_alias: str
    llama_completion_tokens: int
    llama_server_args: list[str]
    llama_readiness_timeout: float
    llama_server_timeout: float
    hormone_model_path: str
    controller_policy_path: str
    affect_classifier_path: str
    hormone_style_map_path: str
    hormone_trace_enabled: bool
    affect_context_enabled: bool
    affect_sampling_preview_enabled: bool
    affect_memory_preview_enabled: bool
    affect_recency_window_seconds: float
    affect_sampling_blend_weight: float
    affect_debug_panel_enabled: bool

    @classmethod
    def load(cls) -> "RuntimeSettings":
        settings = load_settings()

        def getter(key: str, env: str, default: Any = None) -> Any:
            return _get_setting(settings, key, env, default)

        base_temperature = _parse_float(getter("sampling_temperature", "LLAMA_SAMPLING_TEMPERATURE"), 0.7)
        base_top_p = _parse_float(getter("sampling_top_p", "LLAMA_SAMPLING_TOP_P"), 0.9)
        base_frequency_penalty = _parse_float(
            getter("sampling_frequency_penalty", "LLAMA_SAMPLING_FREQUENCY_PENALTY"),
            1.0,
        )
        llm_endpoint = str(getter("llm_endpoint", "LIVING_LLM_URL", "") or "").strip()
        llm_timeout = _parse_float(getter("llm_timeout", "LIVING_LLM_TIMEOUT"), 30.0)
        llama_server_bin = str(getter("llama_server_bin", "LLAMA_SERVER_BIN", "") or "").strip()
        llama_model_path = str(getter("llama_model_path", "LLAMA_MODEL_PATH", "") or "").strip()
        llama_server_host = str(getter("llama_server_host", "LLAMA_SERVER_HOST", "127.0.0.1"))
        llama_server_port = _parse_int(getter("llama_server_port", "LLAMA_SERVER_PORT"), 8080)
        llama_model_alias = str(getter("llama_model_alias", "LLAMA_MODEL_ALIAS", "default"))
        llama_completion_tokens = _parse_int(
            getter("llama_completion_tokens", "LLAMA_COMPLETION_TOKENS"),
            768,
        )
        server_args = getter("llama_server_args", "LLAMA_SERVER_ARGS", "")
        if isinstance(server_args, str):
            llama_server_args: list[str] = shlex.split(server_args)
        elif isinstance(server_args, Sequence):
            llama_server_args = [str(arg) for arg in server_args]
        else:
            llama_server_args = []
        llama_readiness_timeout = _parse_float(
            getter("llama_server_ready_timeout", "LLAMA_SERVER_READY_TIMEOUT"),
            30.0,
        )
        llama_server_timeout = _parse_float(
            getter("llama_server_timeout", "LLAMA_SERVER_TIMEOUT"),
            max(llm_timeout, 60.0),
        )
        hormone_model_path = str(
            getter("hormone_model_path", "HORMONE_MODEL_PATH", "config/hormone_model.json") or ""
        ).strip()
        controller_policy_path = str(
            getter("controller_policy_path", "CONTROLLER_POLICY_PATH", "config/controller_policy.json") or ""
        ).strip()
        affect_classifier_path = str(
            getter("affect_classifier_path", "AFFECT_CLASSIFIER_PATH", "config/affect_classifier.json") or ""
        ).strip()
        hormone_style_map_path = str(
            getter("hormone_style_map_path", "HORMONE_STYLE_MAP_PATH", "config/hormone_style_map.json") or ""
        ).strip()
        hormone_trace_enabled = _parse_bool(
            getter("hormone_trace_enabled", "LIVING_HORMONE_TRACE"),
            True,
        )
        affect_context_enabled = _parse_bool(
            getter("affect_context_enabled", "AFFECT_CONTEXT_ENABLED"),
            False,
        )
        affect_sampling_preview_enabled = _parse_bool(
            getter("affect_sampling_preview_enabled", "AFFECT_SAMPLING_PREVIEW_ENABLED"),
            True,
        )
        affect_memory_preview_enabled = _parse_bool(
            getter("affect_memory_preview_enabled", "AFFECT_MEMORY_PREVIEW_ENABLED"),
            False,
        )
        affect_recency_window_seconds = _parse_float(
            getter("affect_recency_window", "AFFECT_RECENCY_WINDOW"),
            3600.0,
        )
        affect_sampling_blend_weight = _parse_float(
            getter("affect_sampling_blend_weight", "AFFECT_SAMPLING_BLEND_WEIGHT"),
            0.65,
        )
        affect_debug_panel_enabled = _parse_bool(
            getter("affect_debug_panel", "AFFECT_DEBUG_PANEL_ENABLED"),
            False,
        )

        return cls(
            raw=settings,
            base_temperature=base_temperature,
            base_top_p=base_top_p,
            base_frequency_penalty=base_frequency_penalty,
            llm_endpoint=llm_endpoint,
            llm_timeout=llm_timeout,
            llama_server_bin=llama_server_bin,
            llama_model_path=llama_model_path,
            llama_server_host=llama_server_host,
            llama_server_port=llama_server_port,
            llama_model_alias=llama_model_alias,
            llama_completion_tokens=llama_completion_tokens,
            llama_server_args=llama_server_args,
            llama_readiness_timeout=llama_readiness_timeout,
            llama_server_timeout=llama_server_timeout,
            hormone_model_path=hormone_model_path,
            controller_policy_path=controller_policy_path,
            affect_classifier_path=affect_classifier_path,
            hormone_style_map_path=hormone_style_map_path,
            hormone_trace_enabled=hormone_trace_enabled,
            affect_context_enabled=affect_context_enabled,
            affect_sampling_preview_enabled=affect_sampling_preview_enabled,
            affect_memory_preview_enabled=affect_memory_preview_enabled,
            affect_recency_window_seconds=affect_recency_window_seconds,
            affect_sampling_blend_weight=affect_sampling_blend_weight,
            affect_debug_panel_enabled=affect_debug_panel_enabled,
        )


__all__ = ["RuntimeSettings"]


def clear_settings_cache() -> None:
    """Reset the cached settings loader."""
    load_settings.cache_clear()


__all__.append("clear_settings_cache")
