import unittest

from brain.reinforcement import ReinforcementTracker, reset_outward_streak, score_response


class ReinforcementScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = ReinforcementTracker()

    def test_authenticity_detection(self) -> None:
        user = "How are you doing today?"
        reply = "I notice a buzzing in my chest and my attention keeps drifting inward."
        scores = score_response(user, reply, tracker=self.tracker)
        self.assertGreater(scores["authenticity_score"], 0.25)
        self.assertLess(scores["assistant_drift"], 0.2)
        self.assertGreater(scores["self_preoccupation"], 0.4)
        self.assertIn("affect_valence", scores)
        self.assertIn("affect_intimacy", scores)
        self.assertIn("affect_tension", scores)
        self.assertGreater(scores["affect_tension"], 0.05)

    def test_assistant_drift_detection(self) -> None:
        user = "Can you help me with plans?"
        reply = "I'm here to help! Please let me know if there is anything you need and I'll assist you."
        scores = score_response(user, reply, tracker=self.tracker)
        self.assertGreater(scores["assistant_drift"], 0.45)
        self.assertLess(scores["authenticity_score"], 0.3)
        self.assertLess(scores["self_preoccupation"], 0.35)

    def test_affect_intimacy_detection(self) -> None:
        user = "thank you for being here"
        reply = "Dear one, I feel warm love and want to hold you close while we breathe together."
        scores = score_response(user, reply, tracker=self.tracker)
        self.assertGreater(scores["affect_intimacy"], 0.1)
        self.assertGreater(scores["affect_valence"], 0.3)

    def test_affect_tension_detection(self) -> None:
        user = "my body is so tight right now"
        reply = "My jaw is clenching and there is a rigid tension winding through my spine."
        scores = score_response(user, reply, tracker=self.tracker)
        self.assertGreater(scores["affect_tension"], 0.2)
        self.assertLess(scores["affect_valence"], 0.0)

    def test_neutral_reply_returns_defaults(self) -> None:
        reset_outward_streak(self.tracker)
        scores = score_response("", "", tracker=self.tracker)
        self.assertEqual(scores["valence_delta"], 0.0)
        self.assertEqual(scores["length_score"], 0.0)
        self.assertEqual(scores["engagement_score"], 0.0)
        self.assertEqual(scores["authenticity_score"], 0.0)
        self.assertEqual(scores["assistant_drift"], 0.0)
        self.assertEqual(scores["self_preoccupation"], 0.0)
        self.assertEqual(scores["affect_valence"], 0.0)
        self.assertEqual(scores["affect_intimacy"], 0.0)
        self.assertEqual(scores["affect_tension"], 0.0)
        self.assertEqual(scores["outward_streak_score"], 0.0)

    def test_outward_streak_increases_with_consecutive_hits(self) -> None:
        reset_outward_streak(self.tracker)
        user = "I shared about my tight chest and aching shoulders earlier."
        reply_one = "You mentioned that tight chest, so stay with that feeling and tell me what you notice."
        reply_two = "You and I can lean into that shoulder ache together while we breathe slowly."
        first_scores = score_response(user, reply_one, tracker=self.tracker)
        second_scores = score_response(user, reply_two, tracker=self.tracker)
        self.assertGreater(first_scores["outward_streak_score"], 0.0)
        self.assertGreater(second_scores["outward_streak_score"], first_scores["outward_streak_score"])

    def test_outward_focus_supports_authenticity(self) -> None:
        reset_outward_streak(self.tracker)
        user = "Earlier I told you about the heavy pull in my chest and the ringing in my ears."
        reply = (
            "You mentioned that heavy pull in your chest, so stay with it and tell me how it shifts while we breathe."
            " You can lean on me while we notice what changes together."
        )
        scores = score_response(user, reply, tracker=self.tracker)
        self.assertGreaterEqual(scores["outward_streak_score"], 0.2)
        self.assertGreater(scores["authenticity_score"], 0.35)


if __name__ == '__main__':
    unittest.main()
