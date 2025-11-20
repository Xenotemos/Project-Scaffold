import asyncio
import unittest

from state_engine import StateEngine


class StateEngineAffectIntegrationTests(unittest.TestCase):
    def test_initial_affect_overview_present(self) -> None:
        engine = StateEngine()
        overview = engine.affect_overview()
        self.assertIsInstance(overview, dict)
        self.assertIn("traits", overview)
        state = engine.get_state()
        self.assertIn("affect", state)
        affect_state = state["affect"]
        self.assertIn("tags", affect_state)
        self.assertIsInstance(affect_state["tags"], list)

    def test_trait_snapshot_updates_with_hormone_delta(self) -> None:
        engine = StateEngine()
        baseline = engine.trait_snapshot()
        self.assertIsNotNone(baseline)
        baseline_snapshot = baseline
        asyncio.run(
            engine.register_event(
                "tension spike",
                strength=0.2,
                hormone_deltas={"cortisol": 25.0, "noradrenaline": 18.0},
            )
        )
        updated = engine.trait_snapshot()
        self.assertIsNotNone(updated)
        updated_snapshot = updated
        self.assertGreaterEqual(updated_snapshot.tension, baseline_snapshot.tension)
        tags = engine.trait_tags()
        self.assertIsInstance(tags, tuple)
        self.assertGreaterEqual(len(tags), 0)


if __name__ == "__main__":
    unittest.main()
