import unittest

from memory.selector import MemoryCandidate, select_memories, score_memories
from state_engine.affect import TraitSnapshot


class MemorySelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            MemoryCandidate(
                key="calm-journal",
                recency=0.6,
                salience=0.45,
                safety=0.95,
                tags=frozenset({"journal", "grounding"}),
            ),
            MemoryCandidate(
                key="user-praise",
                recency=0.9,
                salience=0.7,
                safety=0.85,
                tags=frozenset({"user", "social"}),
            ),
            MemoryCandidate(
                key="novel-discovery",
                recency=0.4,
                salience=0.92,
                safety=0.6,
                tags=frozenset({"novel", "research"}),
            ),
            MemoryCandidate(
                key="stress-event",
                recency=0.8,
                salience=0.8,
                safety=0.2,
                tags=frozenset({"stress"}),
            ),
        ]

    def test_curiosity_prefers_novel_memories(self) -> None:
        traits = TraitSnapshot(steadiness=0.1, curiosity=0.8, warmth=0.0, tension=-0.1)
        ranked = select_memories(traits, self.candidates, limit=2)
        self.assertEqual(ranked[0].key, "novel-discovery")
        self.assertNotEqual(ranked[-1].key, "stress-event")

    def test_tension_penalizes_risky_memories(self) -> None:
        tense = TraitSnapshot(steadiness=-0.2, curiosity=0.2, warmth=-0.3, tension=0.9)
        scored = score_memories(tense, self.candidates)
        score_map = {candidate.key: score for candidate, score in scored}
        self.assertLess(score_map["stress-event"], score_map["calm-journal"])
        self.assertLess(score_map["stress-event"], score_map["user-praise"])

    def test_warmth_favors_social_memories(self) -> None:
        warm = TraitSnapshot(steadiness=0.2, curiosity=0.2, warmth=0.9, tension=-0.3)
        ranked = select_memories(warm, self.candidates, limit=3)
        keys = [candidate.key for candidate in ranked]
        self.assertIn("user-praise", keys[:2])

    def test_endocrine_spikes_align_with_curiosity(self) -> None:
        dopaminic = MemoryCandidate(
            key="spark-memory",
            recency=0.5,
            salience=0.65,
            safety=0.75,
            tags=frozenset({"internal", "reflection", "spike:dopamine:surging"}),
            spikes={"dopamine": 0.82},
        )
        baseline = MemoryCandidate(
            key="calm-baseline",
            recency=0.7,
            salience=0.6,
            safety=0.9,
            tags=frozenset({"journal"}),
        )
        traits = TraitSnapshot(steadiness=0.1, curiosity=0.85, warmth=0.1, tension=0.2)
        scored = score_memories(
            traits,
            [dopaminic, baseline],
            hormone_bands={"dopamine": "surging"},
        )
        self.assertGreater(scored[0][1], scored[1][1])
        self.assertEqual(scored[0][0].key, "spark-memory")


if __name__ == "__main__":
    unittest.main()
