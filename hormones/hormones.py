"""Hormone system simulating biochemical state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_STIMULUS_DELTAS: dict[str, dict[str, float]] = {
    "reward": {"dopamine": 10.0, "serotonin": 5.0, "noradrenaline": 4.0},
    "stress": {"cortisol": 12.0, "serotonin": -5.0, "noradrenaline": 6.0},
    "affection": {"oxytocin": 10.0, "cortisol": -3.0, "noradrenaline": -4.0},
}


@dataclass(slots=True)
class HormoneLevels:
    """Container for a snapshot of hormone levels."""

    dopamine: float
    serotonin: float
    cortisol: float
    oxytocin: float
    noradrenaline: float

    def as_dict(self) -> dict[str, float]:
        """Expose the hormone levels as a serializable dictionary."""
        return {
            "dopamine": self.dopamine,
            "serotonin": self.serotonin,
            "cortisol": self.cortisol,
            "oxytocin": self.oxytocin,
            "noradrenaline": self.noradrenaline,
        }


class HormoneSystem:
    """Updates hormone levels with decay, stimuli responses, and mood derivation."""

    def __init__(self) -> None:
        self._levels = HormoneLevels(
            dopamine=55.0,
            serotonin=55.0,
            cortisol=35.0,
            oxytocin=45.0,
            noradrenaline=48.0,
        )
        self._baseline = HormoneLevels(
            dopamine=50.0,
            serotonin=50.0,
            cortisol=30.0,
            oxytocin=40.0,
            noradrenaline=45.0,
        )
        self._decay_factors: dict[str, float] = {
            "dopamine": 0.97,
            "serotonin": 0.98,
            "cortisol": 0.94,
            "oxytocin": 0.96,
            "noradrenaline": 0.95,
        }

    def advance(self, seconds: float) -> None:
        """Apply natural decay over the provided interval."""
        decay_exponent = max(seconds / 2, 1.0)
        for name, factor in self._decay_factors.items():
            current = getattr(self._levels, name)
            baseline = getattr(self._baseline, name)
            decayed = baseline + (current - baseline) * (factor ** decay_exponent)
            self._set_level(name, decayed)

    def apply_deltas(self, deltas: Mapping[str, float]) -> None:
        """Adjust hormone levels by the provided delta mapping."""
        for name, delta in deltas.items():
            if hasattr(self._levels, name):
                current = getattr(self._levels, name)
                self._set_level(name, current + float(delta))

    def apply_stimulus(self, stimulus_type: str) -> None:
        """Apply a named stimulus using the predefined delta mappings."""
        try:
            deltas = _STIMULUS_DELTAS[stimulus_type]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(
                f"Unknown stimulus '{stimulus_type}'. Expected one of: "
                f"{', '.join(sorted(_STIMULUS_DELTAS))}"
            ) from exc
        self.apply_deltas(deltas)

    def adjust_baseline(self, adjustments: Mapping[str, float]) -> None:
        """Shift baseline hormone levels incrementally."""
        for name, delta in adjustments.items():
            if hasattr(self._baseline, name):
                current = getattr(self._baseline, name)
                updated = max(0.0, min(100.0, current + float(delta)))
                setattr(self._baseline, name, updated)

    def get_state(self) -> dict[str, float]:
        """Return a read-only snapshot of current hormone levels."""
        return self._levels.as_dict()

    def snapshot(self) -> dict[str, float]:
        """Compatibility alias for get_state."""
        return self.get_state()

    def baseline(self) -> dict[str, float]:
        """Return the baseline hormone levels used for normalization."""
        return self._baseline.as_dict()

    def derive_mood(self) -> str:
        """Derive a coarse mood label from the current hormone balance."""
        state = self.get_state()
        dopamine = state["dopamine"]
        serotonin = state["serotonin"]
        cortisol = state["cortisol"]
        oxytocin = state["oxytocin"]

        if cortisol > 65:
            return "stressed"
        if dopamine > 70 and serotonin > 60:
            return "happy"
        if oxytocin > 60:
            return "affectionate"
        return "neutral"

    def _set_level(self, name: str, value: float) -> None:
        """Clamp hormone levels to the 0-100 range."""
        clamped = max(0.0, min(100.0, value))
        setattr(self._levels, name, clamped)


__all__ = ["HormoneLevels", "HormoneSystem"]
