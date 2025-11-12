"""Sampling policy helpers that translate trait states into model parameters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from state_engine.affect import TraitSnapshot

_TEMP_BOUNDS = (0.4, 1.1)
_TOP_P_BOUNDS = (0.7, 0.98)
_PRESENCE_BOUNDS = (-0.2, 0.8)
_FREQUENCY_BOUNDS = (-0.2, 0.9)
_TOKEN_BOUNDS = (128, 1024)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


@dataclass(frozen=True)
class SamplingPolicy:
    """Model sampling configuration."""

    temperature: float = 0.72
    top_p: float = 0.92
    presence_penalty: float = 0.1
    frequency_penalty: float = 0.2
    max_tokens: int = 512

    def as_kwargs(self) -> Mapping[str, float | int]:
        """Expose the policy as keyword arguments for LLM clients."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
        }


def derive_policy(traits: TraitSnapshot, base: SamplingPolicy | None = None) -> SamplingPolicy:
    """Derive a concrete sampling policy from the supplied traits."""
    policy = base or SamplingPolicy()

    temperature = _clamp(
        policy.temperature + 0.32 * traits.curiosity - 0.24 * traits.tension,
        _TEMP_BOUNDS,
    )
    top_p = _clamp(
        policy.top_p + 0.1 * traits.curiosity - 0.14 * traits.tension,
        _TOP_P_BOUNDS,
    )
    presence_penalty = _clamp(
        policy.presence_penalty
        + 0.35 * traits.curiosity
        + 0.22 * traits.tension
        - 0.12 * traits.steadiness,
        _PRESENCE_BOUNDS,
    )
    frequency_penalty = _clamp(
        policy.frequency_penalty + 0.24 * traits.tension - 0.18 * traits.steadiness,
        _FREQUENCY_BOUNDS,
    )
    token_bonus = int(70 * traits.steadiness - 45 * traits.tension)
    max_tokens = int(_clamp(policy.max_tokens + token_bonus, _TOKEN_BOUNDS))

    return replace(
        policy,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
    )


__all__ = ["SamplingPolicy", "derive_policy"]
