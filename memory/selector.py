"""Memory selection helpers that account for the current trait state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping

from state_engine.affect import TraitSnapshot


@dataclass(frozen=True)
class MemoryCandidate:
    """A lightweight memory descriptor suitable for ranking."""

    key: str
    recency: float
    salience: float
    safety: float
    tags: frozenset[str]
    spikes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.recency <= 1.0:
            raise ValueError("recency must be within [0, 1]")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError("salience must be within [0, 1]")
        if not 0.0 <= self.safety <= 1.0:
            raise ValueError("safety must be within [0, 1]")


def _base_score(candidate: MemoryCandidate) -> float:
    return 0.55 * candidate.salience + 0.35 * candidate.recency + 0.1 * candidate.safety


def score_memories(
    traits: TraitSnapshot,
    candidates: Iterable[MemoryCandidate],
    *,
    hormone_bands: Mapping[str, str] | None = None,
) -> list[tuple[MemoryCandidate, float]]:
    """Score each candidate using trait-aware heuristics."""
    scored: list[tuple[MemoryCandidate, float]] = []
    hormone_bands = hormone_bands or {}
    for candidate in candidates:
        score = _base_score(candidate)

        # Calm states favor coherent, high-safety memories.
        score += 0.15 * traits.steadiness * candidate.safety
        # Curiosity steers toward salient and tagged curiosities.
        if "novel" in candidate.tags:
            score += 0.12 * traits.curiosity
        # Warmth biases toward social/shared memories.
        if candidate.tags & {"social", "mentor", "user"}:
            score += 0.18 * traits.warmth
        # Internal reflections surface when tension or curiosity climb.
        if candidate.tags & {"internal", "reflection"}:
            curiosity_pull = max(traits.curiosity, 0.0) * 0.14
            tension_pull = max(traits.tension, 0.0) * 0.24
            score += curiosity_pull + tension_pull
        if traits.tension > 0:
            risk_penalty = (1.0 - candidate.safety) * (0.35 + 0.25 * candidate.salience)
            score -= traits.tension * risk_penalty

        if candidate.spikes:
            band_lookup: dict[str, str] = {}
            for tag in candidate.tags:
                if tag.startswith("spike:"):
                    parts = tag.split(":")
                    if len(parts) == 3:
                        band_lookup[parts[1]] = parts[2]
            for hormone, magnitude in candidate.spikes.items():
                intensity = abs(magnitude)
                if intensity < 0.35:
                    continue
                direction = 1.0 if magnitude >= 0 else -1.0
                band = band_lookup.get(hormone, "steady")
                current_band = hormone_bands.get(hormone, "steady")
                alignment = 0.0
                if band != "steady" and current_band == band:
                    alignment = 0.12 * intensity
                if hormone == "dopamine":
                    score += (0.18 * max(traits.curiosity, 0.0) * max(intensity, 0.35)) + alignment
                elif hormone == "serotonin":
                    score += (0.14 * max(traits.steadiness, 0.0) * max(intensity, 0.35)) + alignment
                elif hormone == "oxytocin":
                    score += (0.16 * max(traits.warmth, 0.0) * max(intensity, 0.35)) + alignment
                elif hormone == "noradrenaline":
                    score += 0.12 * max(traits.tension, 0.0) * max(intensity, 0.35) + alignment
                elif hormone == "cortisol":
                    if direction > 0:
                        score += 0.22 * max(traits.tension, 0.0) * intensity + alignment
                    else:
                        score += 0.1 * max(0.0, 0.6 - traits.tension) * intensity

        scored.append((candidate, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def select_memories(
    traits: TraitSnapshot,
    candidates: Iterable[MemoryCandidate],
    *,
    limit: int = 4,
    hormone_bands: Mapping[str, str] | None = None,
) -> list[MemoryCandidate]:
    """Return the top memories after trait-aware scoring."""
    ranked = score_memories(traits, candidates, hormone_bands=hormone_bands)
    return [candidate for candidate, _ in ranked[:limit]]


__all__ = ["MemoryCandidate", "score_memories", "select_memories"]
