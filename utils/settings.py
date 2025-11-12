"""Utility helpers for loading optional project settings."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("living_ai.settings")
BASE_DIR = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    """Load settings from the configured settings file if present."""
    env_path = os.getenv("LIVING_SETTINGS_PATH")
    if env_path:
        config_path = Path(env_path)
    else:
        file_name = os.getenv("LIVING_SETTINGS_FILE", "settings.json")
        config_path = BASE_DIR / "config" / file_name
    if not config_path.exists():
        return {}
    try:
        raw_text = config_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception as exc:
        logger.warning("Failed to load settings file %s: %s", config_path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Settings file %s must contain a JSON object.", config_path)
        return {}
    return data


__all__ = ["load_settings"]
