"""Affect integration utilities for transforming hormones into longer-lived traits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

_DEFAULT_HORMONE_BASELINE = 50.0
_DEFAULT_HORMONE_WIDTH = 40.0
_MIN_TRAIT = -1.0
_MAX_TRAIT = 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(value: float, *, midpoint: float, width: float) -> float:
    # Map a hormone value to approximately [-1, 1] around the midpoint.
    if width <= 0:
        raise ValueError("width must be positive")
    return _clamp((value - midpoint) / width, _MIN_TRAIT, _MAX_TRAIT)


def _lerp(previous: float, new: float, factor: float) -> float:
    return previous + factor * (new - previous)


@dataclass(frozen=True)
class AffectState:
    """Fast-changing affect snapshot derived from current hormone levels."""

    dopamine: float
    serotonin: float
    cortisol: float
    oxytocin: float
    noradrenaline: float

    @classmethod
    def from_hormones(
        cls,
        hormones: Mapping[str, float],
        *,
        midpoint: float = _DEFAULT_HORMONE_BASELINE,
    ) -> "AffectState":
        """Build from raw hormone values, defaulting to the midpoint when missing."""
        get = lambda key: float(hormones.get(key, midpoint))
        return cls(
            dopamine=get("dopamine"),
            serotonin=get("serotonin"),
            cortisol=get("cortisol"),
            oxytocin=get("oxytocin"),
            noradrenaline=get("noradrenaline"),
        )

    def as_normalized_vector(
        self,
        *,
        midpoint: float = _DEFAULT_HORMONE_BASELINE,
        width: float = _DEFAULT_HORMONE_WIDTH,
    ) -> tuple[float, float, float, float, float]:
        """Return the hormone values mapped into approximately [-1, 1]."""
        return (
            _normalize(self.dopamine, midpoint=midpoint, width=width),
            _normalize(self.serotonin, midpoint=midpoint, width=width),
            _normalize(self.cortisol, midpoint=midpoint, width=width),
            _normalize(self.oxytocin, midpoint=midpoint, width=width),
            _normalize(self.noradrenaline, midpoint=midpoint, width=width),
        )


@dataclass(frozen=True)
class TraitSnapshot:
    """Slow-changing trait state distilled from affect."""

    steadiness: float
    curiosity: float
    warmth: float
    tension: float

    def dominant_traits(self, *, threshold: float = 0.35) -> list[str]:
        """Return trait labels that stand out above the threshold."""
        traits = {
            "steady": self.steadiness,
            "curious": self.curiosity,
            "warm": self.warmth,
            "tense": self.tension,
        }
        filtered = [name for name, score in traits.items() if score >= threshold]
        return sorted(filtered, key=lambda name: traits[name], reverse=True)

    def reflection_flags(
        self,
        *,
        tension_threshold: float = 0.6,
        steadiness_floor: float = -0.4,
        warmth_floor: float = -0.5,
    ) -> set[str]:
        """Suggest internal reflections that might be warranted."""
        flags: set[str] = set()
        if self.tension >= tension_threshold:
            flags.add("needs_grounding")
        if self.steadiness <= steadiness_floor:
            flags.add("memory_cleanup")
        if self.warmth <= warmth_floor:
            flags.add("social_reconnection")
        return flags


def integrate_traits(
    affect: AffectState,
    *,
    previous: TraitSnapshot | None = None,
    smoothing: float = 0.6,
) -> TraitSnapshot:
    """Blend fast affect inputs into slower traits."""
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be within [0, 1]")

    dop, ser, cor, oxy, nor = affect.as_normalized_vector()

    raw = TraitSnapshot(
        steadiness=_clamp(ser - 0.4 * cor - 0.15 * nor, _MIN_TRAIT, _MAX_TRAIT),
        curiosity=_clamp(dop - 0.25 * cor, _MIN_TRAIT, _MAX_TRAIT),
        warmth=_clamp(oxy + 0.2 * ser - 0.1 * nor, _MIN_TRAIT, _MAX_TRAIT),
        tension=_clamp(cor + 0.35 * nor - 0.1 * ser, _MIN_TRAIT, _MAX_TRAIT),
    )

    if previous is None or smoothing == 0.0:
        return raw

    factor = 1.0 - smoothing
    return TraitSnapshot(
        steadiness=_clamp(_lerp(previous.steadiness, raw.steadiness, factor), _MIN_TRAIT, _MAX_TRAIT),
        curiosity=_clamp(_lerp(previous.curiosity, raw.curiosity, factor), _MIN_TRAIT, _MAX_TRAIT),
        warmth=_clamp(_lerp(previous.warmth, raw.warmth, factor), _MIN_TRAIT, _MAX_TRAIT),
        tension=_clamp(_lerp(previous.tension, raw.tension, factor), _MIN_TRAIT, _MAX_TRAIT),
    )


def traits_to_tags(traits: TraitSnapshot, *, top_k: int = 3, threshold: float = 0.25) -> tuple[str, ...]:
    """Convert prominent traits into lightweight tags suitable for internal use."""
    scored = {
        "steady": traits.steadiness,
        "curious": traits.curiosity,
        "warm": traits.warmth,
        "tense": traits.tension,
    }
    eligible = [(name, score) for name, score in scored.items() if score >= threshold]
    eligible.sort(key=lambda item: item[1], reverse=True)
    return tuple(name for name, _ in eligible[:top_k])


def blend_tags(sequences: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    """Merge multiple tag tuples while preserving order of prominence."""
    seen: set[str] = set()
    merged: list[str] = []
    for seq in sequences:
        for tag in seq:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return tuple(merged)


__all__ = [
    "AffectState",
    "TraitSnapshot",
    "integrate_traits",
    "traits_to_tags",
    "blend_tags",
]
