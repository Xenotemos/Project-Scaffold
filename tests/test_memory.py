"""Unit tests for the memory manager and repository."""

from __future__ import annotations

import tempfile
import unittest

from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from memory import MemoryManager, MemoryRepository
except ModuleNotFoundError as exc:
    MemoryManager = None  # type: ignore[assignment]
    MemoryRepository = None  # type: ignore[assignment]
    MISSING_DEPENDENCY_REASON = str(exc)
else:
    MISSING_DEPENDENCY_REASON = ""


@unittest.skipIf(MemoryManager is None, f"sqlmodel unavailable: {MISSING_DEPENDENCY_REASON}")
class MemoryManagerTests(unittest.TestCase):
    """Confirm MemoryManager consolidation behavior."""

    def _make_repo(self) -> MemoryRepository:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        db_path = Path(temp_dir.name) / "memories.db"
        repository = MemoryRepository(database_path=db_path)
        self.addCleanup(repository.dispose)
        return repository

    def test_event_consolidation(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(
            repository=repository,
            consolidation_interval=timedelta(seconds=0),
            consolidation_threshold=0.6,
        )
        manager.record_event(
            content="test event",
            strength=0.8,
            mood="focused",
            hormone_snapshot={"dopamine": 60.0},
        )
        manager.tick(now=datetime.now(timezone.utc))
        long_term = list(manager.recent_long_term(limit=5))
        self.assertEqual(len(long_term), 1)
        self.assertEqual(long_term[0].content, "test event")
        self.assertGreaterEqual(long_term[0].strength, 0.6)

    def test_decay_prunes_weak_events(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(repository=repository, decay_rate=0.5)
        manager.record_event(
            content="ephemeral",
            strength=0.2,
            mood="focused",
            hormone_snapshot={"dopamine": 40.0},
        )
        manager.tick(now=datetime.now(timezone.utc) + timedelta(seconds=10))
        self.assertEqual(manager.summarize_recent(), "no recent events")

    def test_working_memory_expires_with_time(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(
            repository=repository,
            working_window=timedelta(seconds=30),
            working_limit=3,
        )
        manager.record_event(
            content="first observation",
            strength=0.7,
            mood="curious",
            hormone_snapshot={"dopamine": 55.0},
        )
        manager.record_event(
            content="second observation",
            strength=0.9,
            mood="engaged",
            hormone_snapshot={"dopamine": 60.0},
        )
        snapshot = manager.working_snapshot()
        self.assertEqual(snapshot[0], "second observation")
        self.assertIn("first observation", snapshot)

        future = datetime.now(timezone.utc) + timedelta(seconds=45)
        manager.tick(now=future)
        self.assertEqual(manager.working_snapshot(), [])

    def test_internal_reflections_surface(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(repository=repository)
        manager.record_event(
            content="internal reflection • warming sky",
            strength=0.8,
            mood="neutral",
            hormone_snapshot={"dopamine": 52.0},
            attributes={"tags": ["internal", "reflection"]},
        )
        manager.record_event(
            content="ordinary note",
            strength=0.9,
            mood="neutral",
            hormone_snapshot={"dopamine": 55.0},
        )
        reflections = manager.recent_internal_reflections(limit=2)
        self.assertTrue(reflections)
        self.assertIn("internal reflection", reflections[0])

    def test_active_tags_prioritises_recent_events(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(repository=repository)
        manager.record_event(
            content="grounding cue",
            strength=0.9,
            mood="steady",
            hormone_snapshot={"cortisol": 40.0},
            attributes={"tags": ["internal", "reflection"]},
        )
        manager.record_event(
            content="craving note",
            strength=0.7,
            mood="tense",
            hormone_snapshot={"dopamine": 62.0},
            attributes={"tags": ["craving", "memory_cleanup"]},
        )
        tags = manager.active_tags(limit=3)
        self.assertIn("internal", tags)
        self.assertIn("craving", tags)
        self.assertLessEqual(len(tags), 3)

    def test_endocrine_trace_attached_to_events(self) -> None:
        repository = self._make_repo()
        manager = MemoryManager(repository=repository)
        trace = {
            "normalized": {"dopamine": 0.75},
            "bands": {"dopamine": "surging"},
        }
        controller = {"adjustments": {"temperature_delta": 0.12}}
        manager.record_event(
            content="surge diary",
            strength=0.8,
            mood="curious",
            hormone_snapshot={"dopamine": 70.0},
            endocrine_trace=trace,
            controller_trace=controller,
        )
        event = manager._short_term[0]
        self.assertIn("endocrine", event.attributes)
        self.assertEqual(event.attributes["endocrine"]["bands"]["dopamine"], "surging")
        self.assertIn("controller", event.attributes)
        self.assertEqual(event.attributes["controller"]["adjustments"]["temperature_delta"], 0.12)

if __name__ == "__main__":
    unittest.main()
