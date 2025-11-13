"""Helpers for resolving and selecting runtime configuration profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from app.constants import MODEL_CONFIG_FILES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def available_profiles() -> Mapping[str, str]:
    """Expose the known profile -> settings-file mapping."""
    return dict(MODEL_CONFIG_FILES)


def resolve_settings_file(profile: str) -> str:
    """Translate a profile name into a concrete settings file."""
    key = (profile or "").strip().lower()
    if key in {"", "default"}:
        key = "instruct"
    file_name = MODEL_CONFIG_FILES.get(key)
    if file_name is None:
        raise ValueError(f"Unknown model profile '{profile}'.")
    config_path = CONFIG_DIR / file_name
    if not config_path.exists():
        raise ValueError(f"Config file '{file_name}' is missing.")
    return file_name


def current_settings_file() -> str:
    """Return the active settings file name."""
    return os.getenv("LIVING_SETTINGS_FILE", "settings.json")


def current_profile() -> str:
    """Infer the active profile name from the settings file."""
    file_name = current_settings_file()
    for profile, mapped in MODEL_CONFIG_FILES.items():
        if mapped == file_name:
            return profile
    return "custom"


__all__ = [
    "available_profiles",
    "current_profile",
    "current_settings_file",
    "resolve_settings_file",
]
