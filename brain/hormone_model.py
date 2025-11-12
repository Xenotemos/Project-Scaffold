"""Utilities for loading and applying the learned hormone dynamics model."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

HORMONE_NAMES: Sequence[str] = ["dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"]
REINFORCEMENT_KEYS: Sequence[str] = [
    "valence_delta",
    "length_score",
    "engagement_score",
    "authenticity_score",
    "assistant_drift",
    "self_preoccupation",
]


@dataclass
class HormoneDynamicsModel:
    feature_names: Sequence[str]
    weights: Dict[str, List[float]]
    hormone_scale: float
    feature_index: Dict[str, int]

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "HormoneDynamicsModel":
        feature_names = list(data.get("feature_names") or [])
        hormone_names = list(data.get("hormone_names") or [])
        if hormone_names != list(HORMONE_NAMES):
            raise ValueError("Hormone names in model do not match expected set.")
        raw_weights = data.get("weights") or {}
        weights: Dict[str, List[float]] = {}
        for hormone in HORMONE_NAMES:
            vector = raw_weights.get(hormone)
            if not isinstance(vector, list):
                raise ValueError(f"Model weights for hormone '{hormone}' missing or invalid.")
            weights[hormone] = [float(value) for value in vector]
        normalization = data.get("normalization") or {}
        hormone_scale = float(normalization.get("hormone_scale", 100.0))
        feature_index = {name: idx for idx, name in enumerate(feature_names)}
        return cls(feature_names=feature_names, weights=weights, hormone_scale=hormone_scale, feature_index=feature_index)

    @classmethod
    def from_path(cls, path: Path) -> "HormoneDynamicsModel":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_json(data)

    def _encode_features(
        self,
        hormones: Mapping[str, float],
        reinforcement: Mapping[str, float],
        *,
        intent: str,
        length_label: str,
        profile: str,
    ) -> List[float]:
        vector = [0.0] * len(self.feature_names)
        idx = self.feature_index.get("bias")
        if idx is not None:
            vector[idx] = 1.0
        for hormone in HORMONE_NAMES:
            idx = self.feature_index.get(f"h_pre_{hormone}")
            if idx is not None:
                vector[idx] = float(hormones.get(hormone, 0.0)) / self.hormone_scale
        for key in REINFORCEMENT_KEYS:
            idx = self.feature_index.get(f"reinforcement_{key}")
            if idx is not None:
                vector[idx] = float(reinforcement.get(key, 0.0))
        if intent:
            idx = self.feature_index.get(f"intent::{intent}")
            if idx is not None:
                vector[idx] = 1.0
        if length_label:
            idx = self.feature_index.get(f"length::{length_label}")
            if idx is not None:
                vector[idx] = 1.0
        if profile:
            idx = self.feature_index.get(f"profile::{profile}")
            if idx is not None:
                vector[idx] = 1.0
        return vector

    def predict_delta(
        self,
        hormones: Mapping[str, float],
        reinforcement: Mapping[str, float],
        *,
        intent: str,
        length_label: str,
        profile: str,
        clamp: float = 5.0,
    ) -> Dict[str, float]:
        vector = self._encode_features(hormones, reinforcement, intent=intent, length_label=length_label, profile=profile)
        delta: Dict[str, float] = {}
        for hormone in HORMONE_NAMES:
            weight_vec = self.weights[hormone]
            value = sum(feature * weight for feature, weight in zip(vector, weight_vec))
            if math.isfinite(value):
                if clamp > 0:
                    value = max(-clamp, min(clamp, value))
                delta[hormone] = value
        return delta


def load_model(path: Path) -> HormoneDynamicsModel | None:
    if not path.exists():
        return None
    try:
        return HormoneDynamicsModel.from_path(path)
    except Exception:
        return None
