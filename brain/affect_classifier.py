"""Lightweight affect classifier for user prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence


POSITIVE_WORDS = {
    "appreciate",
    "care",
    "cheerful",
    "connected",
    "gentle",
    "glad",
    "grateful",
    "happy",
    "hope",
    "hopeful",
    "joy",
    "joyful",
    "kind",
    "light",
    "love",
    "loved",
    "lovely",
    "peace",
    "peaceful",
    "playful",
    "proud",
    "relaxed",
    "relief",
    "safe",
    "secure",
    "support",
    "supportive",
    "tender",
    "warm",
}

NEGATIVE_WORDS = {
    "ache",
    "aching",
    "alone",
    "angry",
    "anxious",
    "ashamed",
    "confused",
    "depressed",
    "drained",
    "fear",
    "frightened",
    "frustrated",
    "hurt",
    "lonely",
    "numb",
    "panic",
    "sad",
    "scared",
    "shaky",
    "stressed",
    "tense",
    "tired",
    "upset",
    "worried",
}

AFFECTION_WORDS = {
    "affection",
    "affectionate",
    "love",
    "loved",
    "loving",
    "luv",
    "babe",
    "baby",
    "beloved",
    "caring",
    "cherish",
    "darling",
    "dear",
    "dearie",
    "dearest",
    "honey",
    "sweetie",
    "sweetness",
    "hug",
    "kiss",
    "lover",
    "mwah",
    "sweet",
    "sweetheart",
    "tenderness",
}

DIMINUTIVE_WORDS = {
    "little",
    "tiny",
    "small",
    "softie",
    "kitten",
    "bud",
}

TENSION_WORDS = {
    "ache",
    "aching",
    "clench",
    "clenched",
    "clenching",
    "rigid",
    "strain",
    "strained",
    "stress",
    "stressed",
    "tight",
    "tighten",
    "tightness",
    "tense",
    "tension",
    "wound",
}

EMOJI_WARM = {"❤️", "💕", "💖", "🥰", "😘", "😍"}
EMOJI_STRESS = {"😰", "😟", "😢", "😭", "😔", "😩", "😖", "😤"}

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class AffectClassification:
    """Container for a single user-turn affect classification."""

    valence: float
    intimacy: float
    tension: float
    confidence: float
    tags: tuple[str, ...]
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "valence": round(self.valence, 4),
            "intimacy": round(self.intimacy, 4),
            "tension": round(self.tension, 4),
            "confidence": round(self.confidence, 4),
            "tags": list(self.tags),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class AffectClassifier:
    """Rule-based affect classifier that can be configured via JSON."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._valence_cap = float(config.get("valence_cap", 4.0))
        self._intimacy_cap = float(config.get("intimacy_cap", 4.0))
        self._tension_cap = float(config.get("tension_cap", 4.0))
        self._emoji_weight = float(config.get("emoji_weight", 1.5))
        self._punctuation_bonus = float(config.get("punctuation_bonus", 0.2))
        self._confidence_scale = float(config.get("confidence_scale", 3.0))

    def classify(self, text: str) -> AffectClassification:
        tokens = _tokenise(text)
        lowered = text.lower()
        positive_hits = _count_members(tokens, POSITIVE_WORDS)
        negative_hits = _count_members(tokens, NEGATIVE_WORDS)
        affection_hits = _count_members(tokens, AFFECTION_WORDS)
        diminutive_hits = _count_members(tokens, DIMINUTIVE_WORDS)
        tension_hits = _count_members(tokens, TENSION_WORDS)

        emoji_warm = sum(1 for ch in text if ch in EMOJI_WARM)
        emoji_stress = sum(1 for ch in text if ch in EMOJI_STRESS)
        affection_hits += int(self._emoji_weight * emoji_warm)
        tension_hits += int(self._emoji_weight * emoji_stress)

        exclamation_bonus = self._punctuation_bonus if "!" in text else 0.0

        valence_score = _score_directional(
            positive_hits,
            negative_hits,
            self._valence_cap,
        )
        intimacy_score = min(
            1.0,
            (affection_hits + 0.5 * diminutive_hits) / self._intimacy_cap,
        )
        if affection_hits and len(tokens) <= 3:
            intimacy_score = min(1.0, intimacy_score + 0.18 * affection_hits)
        tension_score = min(
            1.0,
            (tension_hits + max(0.0, -exclamation_bonus)) / self._tension_cap,
        )

        if exclamation_bonus and valence_score > 0:
            valence_score = min(1.0, valence_score + exclamation_bonus)

        signal_strength = (
            abs(valence_score)
            + intimacy_score
            + tension_score
            + (positive_hits + negative_hits + affection_hits + tension_hits) * 0.05
        )
        confidence = max(
            0.0,
            min(1.0, signal_strength / max(1.0, self._confidence_scale)),
        )
        if affection_hits:
            confidence = max(confidence, min(1.0, 0.2 + 0.12 * affection_hits))
        tags = _derive_tags(valence_score, intimacy_score, tension_score, lowered)

        metadata = {
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "affection_hits": affection_hits,
            "tension_hits": tension_hits,
        }
        return AffectClassification(
            valence=valence_score,
            intimacy=intimacy_score,
            tension=tension_score,
            confidence=confidence,
            tags=tuple(tags),
            metadata=metadata,
        )


def load_affect_classifier(config_path: str | Path | None = None) -> AffectClassifier:
    """Load an AffectClassifier from JSON config or fall back to defaults."""
    if not config_path:
        return AffectClassifier()
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return AffectClassifier()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AffectClassifier()
    return AffectClassifier(config)


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _count_members(tokens: Sequence[str], vocab: set[str]) -> int:
    return sum(1 for token in tokens if token in vocab)


def _score_directional(
    positive_hits: int,
    negative_hits: int,
    cap: float,
) -> float:
    numerator = positive_hits - negative_hits
    denominator = max(cap, 1.0)
    return max(-1.0, min(1.0, numerator / denominator))


def _derive_tags(
    valence: float,
    intimacy: float,
    tension: float,
    lowered_text: str,
) -> list[str]:
    tags: list[str] = []
    if valence >= 0.35:
        tags.append("positive")
    elif valence <= -0.35:
        tags.append("negative")
    if intimacy >= 0.2:
        tags.append("affectionate")
    if tension >= 0.2:
        tags.append("tense")
    if "?" in lowered_text:
        tags.append("curious")
    return tags


__all__ = ["AffectClassifier", "AffectClassification", "load_affect_classifier"]
