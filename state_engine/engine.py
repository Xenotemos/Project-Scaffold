"""Core state engine coordinating hormones, affect, and memory."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from hormones import HormoneSystem
from memory import MemoryManager
from .affect import AffectState, TraitSnapshot, integrate_traits, traits_to_tags


class StateEngine:
    """Advance the simulated organism by reconciling hormones and memories."""

    def __init__(
        self,
        *,
        tick_interval: float = 2.0,
        hormone_system: HormoneSystem | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.tick_interval = tick_interval
        self.hormone_system = hormone_system or HormoneSystem()
        self.memory_manager = memory_manager or MemoryManager()
        self._noise_amplitude = 0.35
        self._trait_smoothing = 0.65
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Set baseline state, traits, and seed memory traces."""
        self._tick_counter = 0
        self.state = {}
        self._last_presence = True
        self._trait_snapshot: TraitSnapshot | None = None
        self._trait_tags: tuple[str, ...] = ()
        self._affect_overview: dict[str, Any] = {}
        self.memory_manager.record_event(
            content="system initialized",
            strength=0.6,
            mood=self._derive_mood(),
            hormone_snapshot=self.hormone_system.get_state(),
            endocrine_trace=self._endocrine_trace(),
        )
        self.state = self._compose_state()

    async def tick(self) -> None:
        """Advance hormone and memory subsystems on a fixed cadence."""
        self.hormone_system.advance(self.tick_interval)
        self.memory_manager.tick()
        self._tick_counter += 1
        self._inject_environmental_noise()
        if self._tick_counter % 15 == 0:
            self.memory_manager.record_event(
                content="internal check-in: I pause to notice breath, pulse, and tension without forcing a response",
                strength=0.4,
                mood=self._derive_mood(),
                hormone_snapshot=self.hormone_system.get_state(),
                endocrine_trace=self._endocrine_trace(),
            )
        self.state = self._compose_state()

    def register_event(
        self,
        content: str,
        *,
        strength: float = 0.5,
        stimulus_type: str | None = None,
        hormone_deltas: dict[str, float] | None = None,
        mood: str | None = None,
    ) -> None:
        """Register an external event, updating hormone and memory stores."""
        if stimulus_type is not None:
            self.hormone_system.apply_stimulus(stimulus_type)
        if hormone_deltas:
            self.hormone_system.apply_deltas(hormone_deltas)
        current_mood = mood or self._derive_mood()
        self._apply_sentiment_feedback(content)
        self.memory_manager.record_event(
            content=content,
            strength=strength,
            mood=current_mood,
            hormone_snapshot=self.hormone_system.get_state(),
            endocrine_trace=self._endocrine_trace(),
        )
        self.state = self._compose_state()

    def snapshot(self) -> dict[str, Any]:
        """Return the latest computed state snapshot."""
        return dict(self.state)

    def trait_snapshot(self) -> TraitSnapshot | None:
        """Expose the most recent trait snapshot."""
        return self._trait_snapshot

    def trait_tags(self) -> tuple[str, ...]:
        """Return the latest affect tags."""
        return self._trait_tags

    def affect_overview(self) -> dict[str, Any]:
        """Return a structured affect overview safe for external sharing."""
        if not self._affect_overview:
            return {}
        overview = dict(self._affect_overview)
        overview["tags"] = list(overview.get("tags", []))
        return overview

    def get_state(self) -> dict[str, Any]:
        """Return a simplified state payload suitable for API responses."""
        hormones = self.hormone_system.get_state()
        payload = {
            "mood": self.state.get("mood"),
            "hormones": dict(hormones),
            "timestamp": self.state.get("timestamp"),
        }
        if self._affect_overview:
            payload["affect"] = {
                "tags": list(self._trait_tags),
                "traits": dict(self._affect_overview.get("traits", {})),
            }
        return payload

    def endocrine_snapshot(self) -> dict[str, Any]:
        """Expose the latest endocrine trace for logging or memory coupling."""
        return self._endocrine_trace()

    def reset(self) -> None:
        """Reinitialize hormone, memory, and affect traces without recreating the engine."""
        repository = getattr(self.memory_manager, "_repository", None)
        dispose = getattr(repository, "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass
        self.hormone_system = type(self.hormone_system)()
        self.memory_manager = type(self.memory_manager)()
        self._initialize_state()

    def _compose_state(self) -> dict[str, Any]:
        """Build the broadcastable state payload."""
        mood = self._derive_mood()
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self._update_traits()
        return {
            "mood": mood,
            "hormones": self.hormone_system.get_state(),
            "memory_summary": self.memory_manager.summarize_recent(),
            "working_memory": self.memory_manager.working_snapshot(),
            "timestamp": timestamp,
            "ticks": self._tick_counter,
            "affect": self.affect_overview(),
        }

    def _derive_mood(self) -> str:
        """Compute the mood label using the defined hierarchy."""
        hormones = self.hormone_system.get_state()
        dopamine = hormones["dopamine"]
        serotonin = hormones["serotonin"]
        cortisol = hormones["cortisol"]
        oxytocin = hormones["oxytocin"]

        if cortisol > 65:
            return "stressed"
        if dopamine > 70 and serotonin > 60:
            return "happy"
        if oxytocin > 60:
            return "affectionate"
        return "neutral"

    def _inject_environmental_noise(self) -> None:
        """Apply subtle random noise and time-based fluctuations."""
        noise = lambda scale=1.0: random.uniform(-self._noise_amplitude, self._noise_amplitude) * scale
        deltas = {
            "dopamine": noise(),
            "serotonin": noise(),
            "cortisol": noise(0.8),
            "oxytocin": noise(0.9),
            "noradrenaline": noise(1.1),
        }
        self.hormone_system.apply_deltas(deltas)

    def _apply_sentiment_feedback(self, content: str) -> None:
        """Adjust hormones based on simple sentiment cues from content."""
        text = (content or "").lower()
        positive_tokens = ("thank", "appreciate", "calm", "glad", "hopeful")
        negative_tokens = ("stressed", "frustrated", "tired", "worried", "angry")
        delta: dict[str, float] = {}
        if any(token in text for token in positive_tokens):
            delta["serotonin"] = delta.get("serotonin", 0.0) + 0.8
            delta["oxytocin"] = delta.get("oxytocin", 0.0) + 0.6
        if any(token in text for token in negative_tokens):
            delta["cortisol"] = delta.get("cortisol", 0.0) + 1.2
            delta["dopamine"] = delta.get("dopamine", 0.0) - 0.7
        if delta:
            self.hormone_system.apply_deltas(delta)

    def _update_traits(self) -> None:
        """Refresh the cached trait snapshot based on the latest hormones."""
        hormones = self.hormone_system.get_state()
        affect = AffectState.from_hormones(hormones)
        snapshot = integrate_traits(
            affect,
            previous=self._trait_snapshot,
            smoothing=self._trait_smoothing,
        )
        tags = traits_to_tags(snapshot)
        self._trait_snapshot = snapshot
        self._trait_tags = tags
        self._affect_overview = {
            "traits": {
                "steadiness": round(snapshot.steadiness, 4),
                "curiosity": round(snapshot.curiosity, 4),
                "warmth": round(snapshot.warmth, 4),
                "tension": round(snapshot.tension, 4),
            },
            "reflection_flags": sorted(snapshot.reflection_flags()),
            "tags": list(tags),
        }

    def _endocrine_trace(self) -> dict[str, Any]:
        """Return hormone levels with baseline deltas and qualitative bands."""
        hormones = self.hormone_system.get_state()
        baseline = self.hormone_system.baseline()
        trace: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "hormones": dict(hormones),
            "baseline": dict(baseline),
            "delta": {},
            "normalized": {},
            "bands": {},
        }
        for name, value in hormones.items():
            base = float(baseline.get(name, 50.0))
            delta = float(value) - base
            normalized = max(-1.5, min(1.5, delta / 45.0))
            trace["delta"][name] = round(delta, 4)
            trace["normalized"][name] = round(normalized, 4)
            trace["bands"][name] = self._classify_band(delta)
        return trace

    @staticmethod
    def _classify_band(delta: float) -> str:
        """Classify hormone delta into qualitative bands."""
        if delta >= 12.0:
            return "surging"
        if delta >= 6.0:
            return "rising"
        if delta <= -12.0:
            return "crashing"
        if delta <= -6.0:
            return "fading"
        return "steady"


__all__ = ["StateEngine"]
