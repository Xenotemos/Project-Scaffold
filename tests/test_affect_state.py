import unittest

from state_engine.affect import (
    AffectState,
    TraitSnapshot,
    blend_tags,
    integrate_traits,
    traits_to_tags,
)


class AffectIntegrationTests(unittest.TestCase):
    def test_affect_state_normalization(self) -> None:
        hormones = {
            "dopamine": 80.0,
            "serotonin": 72.0,
            "cortisol": 45.0,
            "oxytocin": 65.0,
            "noradrenaline": 40.0,
        }
        affect = AffectState.from_hormones(hormones)
        normalized = affect.as_normalized_vector()
        self.assertEqual(len(normalized), 5)
        # Dopamine and serotonin are above baseline.
        self.assertGreater(normalized[0], 0.0)
        self.assertGreater(normalized[1], 0.0)
        # Cortisol is slightly below baseline.
        self.assertLess(normalized[2], 0.0)

    def test_trait_integration_with_smoothing(self) -> None:
        baseline = TraitSnapshot(steadiness=0.0, curiosity=0.0, warmth=0.0, tension=0.0)
        affect_high_tension = AffectState.from_hormones(
            {
                "dopamine": 40.0,
                "serotonin": 42.0,
                "cortisol": 90.0,
                "oxytocin": 35.0,
                "noradrenaline": 88.0,
            }
        )
        first = integrate_traits(affect_high_tension, previous=baseline, smoothing=0.5)
        self.assertLess(first.steadiness, 0.0)
        self.assertGreaterEqual(first.tension, 0.5)

        affect_recovering = AffectState.from_hormones(
            {
                "dopamine": 60.0,
                "serotonin": 68.0,
                "cortisol": 48.0,
                "oxytocin": 70.0,
                "noradrenaline": 40.0,
            }
        )
        second = integrate_traits(affect_recovering, previous=first, smoothing=0.6)
        self.assertGreater(second.steadiness, first.steadiness)
        self.assertLessEqual(second.tension, first.tension)

    def test_traits_to_tags(self) -> None:
        traits = TraitSnapshot(steadiness=0.6, curiosity=0.2, warmth=0.5, tension=-0.1)
        tags = traits_to_tags(traits, threshold=0.25)
        self.assertEqual(tags, ("steady", "warm"))
        merged = blend_tags([tags, ("curious",), ("steady",)])
        self.assertEqual(merged, ("steady", "warm", "curious"))

    def test_reflection_flags(self) -> None:
        tense = TraitSnapshot(steadiness=-0.6, curiosity=0.0, warmth=-0.6, tension=0.8)
        self.assertEqual(
            tense.reflection_flags(),
            {"needs_grounding", "memory_cleanup", "social_reconnection"},
        )


if __name__ == "__main__":
    unittest.main()
