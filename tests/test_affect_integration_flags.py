import importlib
import types
import unittest
from datetime import datetime, timezone, timedelta

from state_engine.affect import TraitSnapshot


class AffectIntegrationFlagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.main = importlib.reload(importlib.import_module("main"))
        self.original_recent = self.main.state_engine.memory_manager.recent_long_term
        self.main.AFFECT_CONTEXT_ENABLED = False
        self.main.AFFECT_SAMPLING_PREVIEW_ENABLED = False
        self.main.AFFECT_MEMORY_PREVIEW_ENABLED = False
        self.main.AFFECT_DEBUG_PANEL_ENABLED = False

    def tearDown(self) -> None:
        self.main.state_engine.memory_manager.recent_long_term = self.original_recent
        self.main.AFFECT_CONTEXT_ENABLED = False
        self.main.AFFECT_SAMPLING_PREVIEW_ENABLED = False
        self.main.AFFECT_MEMORY_PREVIEW_ENABLED = False
        self.main.AFFECT_DEBUG_PANEL_ENABLED = False

    def test_sampling_policy_blend_modifies_sampling(self) -> None:
        message = "Explore this idea for a moment."
        self.main.state_engine._trait_snapshot = TraitSnapshot(steadiness=0.4, curiosity=0.9, warmth=0.1, tension=-0.2)
        self.main.AFFECT_SAMPLING_PREVIEW_ENABLED = False
        _, _, _, baseline_sampling, _ = self.main._prepare_chat_request(message)

        self.main.AFFECT_SAMPLING_PREVIEW_ENABLED = True
        self.main.state_engine._trait_snapshot = TraitSnapshot(steadiness=0.4, curiosity=0.9, warmth=0.1, tension=-0.2)
        context, _, _, blended_sampling, _ = self.main._prepare_chat_request(message)

        self.assertIn("sampling_policy_preview", context)
        self.assertNotAlmostEqual(baseline_sampling["temperature"], blended_sampling["temperature"])
        self.assertNotAlmostEqual(baseline_sampling["top_p"], blended_sampling["top_p"])

    def test_memory_reordering_applies_preview_ranking(self) -> None:
        Record = types.SimpleNamespace
        now = datetime.now(timezone.utc)
        records = [
            Record(
                id=1,
                content="calm note",
                strength=0.8,
                mood="calm",
                attributes={"safety": 0.95, "tags": ["journal"]},
                created_at=now - timedelta(minutes=10),
            ),
            Record(
                id=2,
                content="stress spike",
                strength=0.9,
                mood="stressed",
                attributes={"safety": 0.2, "tags": ["stress"]},
                created_at=now - timedelta(minutes=5),
            ),
            Record(
                id=3,
                content="novel idea",
                strength=0.7,
                mood="curious",
                attributes={"safety": 0.6, "tags": ["novel"]},
                created_at=now - timedelta(minutes=2),
            ),
        ]
        self.main.state_engine.memory_manager.recent_long_term = lambda limit=5: records  # type: ignore[assignment]
        self.main.AFFECT_CONTEXT_ENABLED = True
        self.main.AFFECT_MEMORY_PREVIEW_ENABLED = True
        self.main.state_engine._trait_snapshot = TraitSnapshot(
            steadiness=-0.2, curiosity=0.1, warmth=0.2, tension=0.85
        )

        context = self.main._build_chat_context()
        memory = context["memory"]
        preview = memory.get("affect_preview")
        long_term = memory.get("long_term")

        self.assertIsInstance(preview, list)
        self.assertEqual(preview[0]["key"], "lt-1")
        self.assertIsInstance(long_term, list)
        self.assertEqual(long_term[0]["content"], "calm note")
        self.assertNotEqual(long_term[1]["content"], "calm note")

    def test_internal_reflections_surface_in_context(self) -> None:
        self.main.state_engine.memory_manager.record_event(
            content="internal reflection • tracing a low hum",
            strength=0.82,
            mood="neutral",
            hormone_snapshot={"dopamine": 51.0},
            attributes={"tags": ["internal", "reflection"]},
        )
        context = self.main._build_chat_context()
        memory_block = context["memory"]
        reflections = memory_block.get("internal_reflections")
        self.assertIsInstance(reflections, list)
        self.assertTrue(reflections)
        self.assertIn("internal reflection", reflections[0])
        self.assertIn("inner_reflections", context)


if __name__ == "__main__":
    unittest.main()
