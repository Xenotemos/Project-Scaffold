"""Unit tests for the hormone system."""

from __future__ import annotations

import unittest

from hormones import HormoneSystem


class HormoneSystemTests(unittest.TestCase):
    """Validate decay and stimulus handling for HormoneSystem."""

    def test_decay_reduces_levels(self) -> None:
        system = HormoneSystem()
        initial_snapshot = system.snapshot()
        system.advance(seconds=4.0)
        decayed_snapshot = system.snapshot()
        for hormone, initial_value in initial_snapshot.items():
            with self.subTest(hormone=hormone):
                self.assertLess(decayed_snapshot[hormone], initial_value)
                self.assertGreaterEqual(decayed_snapshot[hormone], 0.0)

    def test_apply_deltas_clamps_and_adjusts(self) -> None:
        system = HormoneSystem()
        system.apply_deltas({"dopamine": 60.0, "oxytocin": 60.0, "noradrenaline": 60.0})
        snapshot = system.get_state()
        self.assertEqual(snapshot["dopamine"], 100.0)
        self.assertEqual(snapshot["oxytocin"], 100.0)
        self.assertEqual(snapshot["noradrenaline"], 100.0)

    def test_apply_stimulus_keyword(self) -> None:
        system = HormoneSystem()
        baseline = system.get_state()
        system.apply_stimulus("reward")
        updated = system.get_state()
        self.assertGreater(updated["dopamine"], baseline["dopamine"])
        self.assertGreater(updated["serotonin"], baseline["serotonin"])

    def test_mood_derivation_hierarchy(self) -> None:
        stressed = HormoneSystem()
        stressed.apply_deltas({"cortisol": 40.0})
        self.assertEqual(stressed.derive_mood(), "stressed")

        happy = HormoneSystem()
        happy.apply_deltas({"dopamine": 30.0, "serotonin": 10.0})
        self.assertEqual(happy.derive_mood(), "happy")

        affectionate = HormoneSystem()
        affectionate.apply_deltas({"oxytocin": 20.0})
        self.assertEqual(affectionate.derive_mood(), "affectionate")

        neutral = HormoneSystem()
        self.assertEqual(neutral.derive_mood(), "neutral")


if __name__ == "__main__":
    unittest.main()
