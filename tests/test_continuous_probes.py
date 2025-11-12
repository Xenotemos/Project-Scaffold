"""Unit tests for the continuous probes helper utilities."""

from __future__ import annotations

import unittest

from scripts import continuous_probes


class ContinuousProbeHelpersTests(unittest.TestCase):
    def test_aggregate_hormones(self) -> None:
        rows = [
            {"hormones": {"dopamine": 60.0, "serotonin": 55.0}},
            {"hormones": {"dopamine": 62.0, "serotonin": 53.0}},
            {"hormones": {"dopamine": 58.0}},
        ]
        aggregates = continuous_probes._aggregate_hormones(rows)
        self.assertAlmostEqual(aggregates["dopamine"], 60.0, places=3)
        self.assertAlmostEqual(aggregates["serotonin"], 54.0, places=3)

    def test_controller_aggregation(self) -> None:
        rows = [
            {"controller": {"applied": {"temperature": 0.1}}},
            {"controller": {"applied": {"temperature": 0.2}}},
            {"controller": {"applied": {"temperature": -0.1}}},
        ]
        aggregates = continuous_probes._aggregate_controller(rows)
        self.assertAlmostEqual(aggregates["temperature"], 0.0666, places=3)

    def test_evaluate_gate_flags_thresholds(self) -> None:
        metrics = {
            "authenticity_score": 0.5,
            "assistant_drift": 0.3,
            "self_preoccupation": 0.5,
        }
        gate = continuous_probes._evaluate_gate(metrics)
        self.assertTrue(gate["authenticity_score"]["ok"])
        self.assertTrue(gate["assistant_drift"]["ok"])
        self.assertTrue(gate["self_preoccupation"]["ok"])
        self.assertTrue(gate["eligible_for_promotion"])


if __name__ == "__main__":
    unittest.main()
