"""Runtime support for the recurrent controller policy that steers sampling."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

logger = logging.getLogger("living_ai.controller_policy")


def _reshape(vector: Sequence[float], rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
    if len(vector) != rows * cols:
        raise ValueError(f"Expected {rows * cols} values, received {len(vector)}.")
    matrix = []
    index = 0
    for _ in range(rows):
        row = [float(vector[index + offset]) for offset in range(cols)]
        matrix.append(tuple(row))
        index += cols
    return tuple(matrix)


def _as_tuple(sequence: Sequence[float], expected_length: int, *, name: str) -> tuple[float, ...]:
    if len(sequence) != expected_length:
        raise ValueError(f"{name} length mismatch (expected {expected_length}, got {len(sequence)}).")
    return tuple(float(value) for value in sequence)


@dataclass(frozen=True)
class ControllerStepResult:
    """Snapshot of a single controller evaluation."""

    adjustments: Dict[str, float]
    hidden_state: tuple[float, ...]
    input_vector: tuple[float, ...]
    raw_outputs: tuple[float, ...]


@dataclass(frozen=True)
class ControllerPolicy:
    """Container describing the trained recurrent controller."""

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    hidden_size: int
    state_decay: float
    output_scales: Dict[str, float]
    output_bounds: Dict[str, tuple[float | None, float | None]]
    weights_input: tuple[tuple[float, ...], ...]
    weights_hidden: tuple[tuple[float, ...], ...]
    bias_hidden: tuple[float, ...]
    weights_output: tuple[tuple[float, ...], ...]
    bias_output: tuple[float, ...]
    metadata: Dict[str, Any]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ControllerPolicy":
        input_names = tuple(str(name) for name in payload.get("input_features") or ())
        output_names = tuple(str(name) for name in payload.get("output_names") or ())
        hidden_size = int(payload.get("hidden_size") or 0)
        if not input_names:
            raise ValueError("Controller policy is missing input feature names.")
        if not output_names:
            raise ValueError("Controller policy is missing output names.")
        if hidden_size <= 0:
            raise ValueError("Controller policy must define a positive hidden size.")

        weights = payload.get("weights") or {}
        input_weights = weights.get("input") or []
        recurrent_weights = weights.get("recurrent") or []
        hidden_bias = weights.get("hidden_bias") or []
        output_weights = weights.get("output") or []
        output_bias = weights.get("output_bias") or []

        weights_input = _reshape(input_weights, hidden_size, len(input_names))
        weights_hidden = _reshape(recurrent_weights, hidden_size, hidden_size)
        bias_hidden = _as_tuple(hidden_bias, hidden_size, name="hidden_bias")
        weights_output = _reshape(output_weights, len(output_names), hidden_size)
        bias_output = _as_tuple(output_bias, len(output_names), name="output_bias")

        state_decay = float(payload.get("state_decay", 1.0))
        state_decay = max(0.0, min(state_decay, 1.0))

        output_scales_raw = payload.get("output_scales") or {}
        output_scales: Dict[str, float] = {}
        for name in output_names:
            output_scales[name] = float(output_scales_raw.get(name, 1.0))

        bounds_raw = payload.get("output_bounds") or {}
        output_bounds: Dict[str, tuple[float | None, float | None]] = {}
        for name in output_names:
            entry = bounds_raw.get(name) or []
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                low = entry[0] if entry[0] is not None else None
                high = entry[1] if entry[1] is not None else None
            else:
                low = None
                high = None
            output_bounds[name] = (float(low) if low is not None else None, float(high) if high is not None else None)

        metadata = dict(payload.get("metadata") or {})

        return cls(
            input_names=input_names,
            output_names=output_names,
            hidden_size=hidden_size,
            state_decay=state_decay,
            output_scales=output_scales,
            output_bounds=output_bounds,
            weights_input=weights_input,
            weights_hidden=weights_hidden,
            bias_hidden=bias_hidden,
            weights_output=weights_output,
            bias_output=bias_output,
            metadata=metadata,
        )

    def runtime(self) -> "ControllerPolicyRuntime":
        return ControllerPolicyRuntime(self)


class ControllerPolicyRuntime:
    """Evaluate the recurrent controller against live features."""

    def __init__(self, policy: ControllerPolicy) -> None:
        self._policy = policy
        self._state: list[float] = [0.0] * policy.hidden_size
        self._last_result: ControllerStepResult | None = None

    def reset(self) -> None:
        """Reset the recurrent state to zero."""
        self._state = [0.0] * self._policy.hidden_size
        self._last_result = None

    @property
    def last_result(self) -> ControllerStepResult | None:
        """Return the most recent step result."""
        return self._last_result

    def step(
        self,
        feature_values: Mapping[str, float] | None = None,
        *,
        tags: Iterable[str] | None = None,
    ) -> ControllerStepResult:
        """Advance the controller using the supplied features."""
        feature_values = feature_values or {}
        tag_set = {str(tag).lower() for tag in (tags or [])}
        input_vector: list[float] = []
        for name in self._policy.input_names:
            if name == "bias":
                input_vector.append(float(feature_values.get(name, 1.0)))
            elif name.startswith("tag:"):
                tag = name.split(":", 1)[1].strip().lower()
                input_vector.append(1.0 if tag in tag_set else 0.0)
            else:
                input_vector.append(float(feature_values.get(name, 0.0)))

        hidden_pre: list[float] = []
        for idx in range(self._policy.hidden_size):
            total = self._policy.bias_hidden[idx]
            total += sum(
                weight * value for weight, value in zip(self._policy.weights_input[idx], input_vector)
            )
            total += sum(
                weight * state_value
                for weight, state_value in zip(self._policy.weights_hidden[idx], self._state)
            )
            hidden_pre.append(math.tanh(total))

        if self._policy.state_decay <= 0.0:
            updated_state = hidden_pre
        elif self._policy.state_decay >= 1.0:
            updated_state = hidden_pre
        else:
            decay = self._policy.state_decay
            updated_state = [
                (1.0 - decay) * previous + decay * current
                for previous, current in zip(self._state, hidden_pre)
            ]

        raw_outputs: list[float] = []
        for row in self._policy.weights_output:
            total = sum(weight * value for weight, value in zip(row, updated_state))
            total += self._policy.bias_output[len(raw_outputs)]
            raw_outputs.append(math.tanh(total))

        adjustments: Dict[str, float] = {}
        for name, raw_value in zip(self._policy.output_names, raw_outputs):
            scaled = raw_value * self._policy.output_scales.get(name, 1.0)
            low, high = self._policy.output_bounds.get(name, (None, None))
            if low is not None:
                scaled = max(low, scaled)
            if high is not None:
                scaled = min(high, scaled)
            adjustments[name] = scaled

        self._state = list(updated_state)
        result = ControllerStepResult(
            adjustments=adjustments,
            hidden_state=tuple(updated_state),
            input_vector=tuple(input_vector),
            raw_outputs=tuple(raw_outputs),
        )
        self._last_result = result
        return result


def load_controller_policy(path: Path) -> ControllerPolicy | None:
    """Load a controller policy from a JSON file."""
    if not path.exists():
        logger.info("Controller policy file %s not found.", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive I/O handling
        logger.warning("Failed to read controller policy file %s: %s", path, exc)
        return None
    try:
        return ControllerPolicy.from_json(payload)
    except Exception as exc:
        logger.warning("Controller policy at %s is invalid: %s", path, exc)
        return None


__all__ = [
    "ControllerPolicy",
    "ControllerPolicyRuntime",
    "ControllerStepResult",
    "load_controller_policy",
]
