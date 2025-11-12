"""High-level memory management and consolidation logic."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Iterable

from .models import LongTermMemoryRecord
from .repository import MemoryRepository


@dataclass(slots=True)
class MemoryEvent:
    """Represents an in-memory snapshot of a recent experience."""

    content: str
    strength: float
    mood: str
    hormone_snapshot: dict[str, float]
    attributes: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def decay(self, rate: float) -> None:
        """Reduce the memory strength by a decay multiplier."""
        self.strength = max(0.0, self.strength * rate)


class MemoryManager:
    """Orchestrates short, working, and long-term memory stores."""

    def __init__(
        self,
        repository: MemoryRepository | None = None,
        short_term_limit: int = 32,
        decay_rate: float = 0.9,
        consolidation_interval: timedelta = timedelta(seconds=30),
        consolidation_threshold: float = 0.7,
        working_window: timedelta = timedelta(seconds=120),
        working_limit: int = 6,
    ) -> None:
        self._repository = repository or MemoryRepository()
        self._short_term: Deque[MemoryEvent] = deque(maxlen=short_term_limit)
        self._working_memory: list[str] = []
        self._decay_rate = decay_rate
        self._consolidation_interval = consolidation_interval
        self._consolidation_threshold = consolidation_threshold
        self._last_consolidation = datetime.now(timezone.utc)
        self._working_window = working_window
        self._working_limit = working_limit

    def record_event(
        self,
        content: str,
        strength: float,
        mood: str,
        hormone_snapshot: dict[str, float],
        *,
        attributes: dict[str, object] | None = None,
        endocrine_trace: dict[str, Any] | None = None,
        controller_trace: dict[str, Any] | None = None,
    ) -> None:
        """Store a new memory event in short-term storage."""
        event = MemoryEvent(
            content=content,
            strength=min(max(strength, 0.0), 1.0),
            mood=mood,
            hormone_snapshot=hormone_snapshot,
            attributes=dict(attributes or {}),
        )
        if endocrine_trace:
            existing = event.attributes.setdefault("endocrine", {})
            if isinstance(existing, dict):
                existing.update(endocrine_trace)
            else:
                event.attributes["endocrine"] = dict(endocrine_trace)
        if controller_trace:
            event.attributes["controller"] = dict(controller_trace)
        self._short_term.appendleft(event)
        self._refresh_working_memory(now=event.created_at)

    def tick(self, now: datetime | None = None) -> None:
        """Advance internal memory timers, decaying and consolidating as needed."""
        now = now or datetime.now(timezone.utc)
        self._apply_decay()
        if now - self._last_consolidation >= self._consolidation_interval:
            self._consolidate(now)
            self._last_consolidation = now
        self._refresh_working_memory(now=now)

    def summarize_recent(self, limit: int = 3) -> str:
        """Summarize the most recent short-term memories."""
        if not self._short_term:
            return "no recent events"
        snippets = [event.content for event in list(self._short_term)[:limit]]
        return ", ".join(snippets)

    def working_snapshot(self) -> list[str]:
        """Expose a shallow copy of the working memory buffer."""
        return list(self._working_memory)

    def recent_long_term(self, limit: int = 5) -> Iterable[LongTermMemoryRecord]:
        """Retrieve recent consolidated memories from persistence."""
        return self._repository.recent(limit=limit)

    def recent_internal_reflections(self, limit: int = 3) -> list[str]:
        """Return recent memory snippets tagged as internal reflections."""
        reflections: list[str] = []

        def _has_tags(tag_source: Iterable[str] | None) -> bool:
            if not tag_source:
                return False
            lowered = {tag.lower() for tag in tag_source}
            return "internal" in lowered or "reflection" in lowered

        for event in self._short_term:
            tags = event.attributes.get("tags") if isinstance(event.attributes, dict) else None
            if _has_tags(tags):
                reflections.append(event.content)
            if len(reflections) >= limit:
                return reflections[:limit]

        recent_records = self._repository.recent(limit=limit * 2)
        for record in recent_records:
            tags = record.attributes.get("tags") if isinstance(record.attributes, dict) else None
            if _has_tags(tags):
                reflections.append(record.content)
            if len(reflections) >= limit:
                break
        return reflections[:limit]

    def active_tags(self, limit: int = 6) -> list[str]:
        """Return prominent tags observed in recent memory activity."""
        if limit <= 0:
            return []
        counts: Counter[str] = Counter()

        def _collect_tags(source: Iterable[str] | None) -> None:
            if not source:
                return
            for tag in source:
                if isinstance(tag, str):
                    cleaned = tag.strip().lower()
                    if cleaned:
                        counts[cleaned] += 1

        for event in self._short_term:
            attributes = event.attributes if isinstance(event.attributes, dict) else {}
            _collect_tags(attributes.get("tags") if isinstance(attributes, dict) else None)
            if len(counts) >= limit:
                break

        if counts:
            return [tag for tag, _ in counts.most_common(limit)]

        for record in self._repository.recent(limit=limit * 2):
            attributes = record.attributes if isinstance(record.attributes, dict) else {}
            _collect_tags(attributes.get("tags") if isinstance(attributes, dict) else None)
            if len(counts) >= limit:
                break

        return [tag for tag, _ in counts.most_common(limit)]

    def _apply_decay(self) -> None:
        """Decay each short-term memory and prune weak ones."""
        for event in list(self._short_term):
            event.decay(self._decay_rate)
            if event.strength <= 0.1:
                self._short_term.remove(event)

    def _consolidate(self, timestamp: datetime) -> None:
        """Persist strong memories to long-term storage."""
        strong_events = [event for event in self._short_term if event.strength >= self._consolidation_threshold]
        for event in strong_events:
            attributes = {"source": "short_term", "captured_at": event.created_at.isoformat()}
            attributes.update(event.attributes)
            record = LongTermMemoryRecord(
                content=event.content,
                mood=event.mood,
                hormone_snapshot=event.hormone_snapshot,
                strength=event.strength,
                attributes=attributes,
                created_at=timestamp,
            )
            self._repository.save_record(record)
            event.strength = min(event.strength, 0.5)

    def _refresh_working_memory(self, *, now: datetime | None = None) -> None:
        """Rebuild the working-memory buffer using a time-based window."""
        reference_time = now or datetime.now(timezone.utc)
        window = self._working_window
        if window.total_seconds() <= 0:
            self._working_memory = []
            return
        fresh_events: list[str] = []
        for event in self._short_term:
            age = reference_time - event.created_at
            if age <= window:
                fresh_events.append(event.content)
        self._working_memory = fresh_events[: self._working_limit]
