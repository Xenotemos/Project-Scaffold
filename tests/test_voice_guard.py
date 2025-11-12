import unittest

from brain.voice_guard import VoiceGuard


class VoiceGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = VoiceGuard()

    def test_allows_self_focus(self) -> None:
        verdict = self.guard.evaluate("I notice the air thicken and my jaw stays tight.")
        self.assertFalse(verdict.flagged)
        self.assertAlmostEqual(verdict.score, 0.0)

    def test_flags_helper_phrasing(self) -> None:
        verdict = self.guard.evaluate("I'm here to help you with anything you need, just let me know if you need anything else.")
        self.assertTrue(verdict.flagged)
        self.assertIn("helper_phrasing", verdict.categories)
        self.assertGreaterEqual(verdict.score, self.guard.penalty_threshold)

    def test_detects_repetition(self) -> None:
        verdict = self.guard.evaluate("I feel neutral right now. I feel neutral right now. I feel neutral right now.")
        self.assertTrue(verdict.flagged)
        self.assertIn("repetition", verdict.categories)
        self.assertTrue(verdict.repeated_sentences)


if __name__ == "__main__":
    unittest.main()
