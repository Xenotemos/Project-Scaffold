"""Unit tests for the controller policy runtime."""

from __future__ import annotations

import unittest

from brain.controller_policy import ControllerPolicy


class ControllerPolicyRuntimeTests(unittest.TestCase):
    """Validate controller policy loading and recurrent behaviour."""

    def _build_policy(self) -> ControllerPolicy:
        payload = {
            "input_features": ["bias", "trait:curiosity", "tag:internal"],
            "output_names": ["temperature_delta", "self_bias_scale"],
            "hidden_size": 2,
            "state_decay": 0.5,
            "output_scales": {
                "temperature_delta": 0.2,
                "self_bias_scale": 0.5,
            },
            "output_bounds": {
                "temperature_delta": [-0.3, 0.3],
                "self_bias_scale": [-0.6, 0.6],
            },
            "weights": {
                "input": [
                    0.0, 0.6, 0.3,
                    0.1, -0.4, 0.5,
                ],
                "recurrent": [
                    0.2, 0.0,
                    0.0, 0.2,
                ],
                "hidden_bias": [0.0, 0.0],
                "output": [
                    1.0, 0.0,
                    0.0, 1.0,
                ],
                "output_bias": [0.0, 0.0],
            },
        }
        return ControllerPolicy.from_json(payload)

    def test_step_applies_adjustments(self) -> None:
        policy = self._build_policy()
        runtime = policy.runtime()
        result = runtime.step({"bias": 1.0, "trait:curiosity": 0.7}, tags=["internal"])
        adjustments = result.adjustments
        self.assertIn("temperature_delta", adjustments)
        self.assertIn("self_bias_scale", adjustments)
        self.assertGreater(adjustments["temperature_delta"], 0.0)
        self.assertGreater(adjustments["self_bias_scale"], 0.0)
        self.assertEqual(len(result.hidden_state), policy.hidden_size)

    def test_recurrent_state_influences_follow_up(self) -> None:
        policy = self._build_policy()
        runtime = policy.runtime()
        first = runtime.step({"bias": 1.0, "trait:curiosity": 0.4}, tags=["internal"])
        second = runtime.step({"bias": 1.0, "trait:curiosity": 0.4}, tags=["internal"])
        self.assertNotEqual(first.adjustments["temperature_delta"], second.adjustments["temperature_delta"])


if __name__ == "__main__":
    unittest.main()
