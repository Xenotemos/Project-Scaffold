"""Helpers for building chat context and scoring memory previews."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from memory.selector import MemoryCandidate, score_memories, select_memories
from state_engine import StateEngine, TraitSnapshot

try:  # pragma: no cover - optional type info
    from brain.local_llama_engine import LocalLlamaEngine
except Exception:  # pragma: no cover
    LocalLlamaEngine = Any  # type: ignore[misc]


def _memory_key(record: Any, index: int) -> str:
    identifier = getattr(record, "id", None)
    if identifier in (None, ""):
        identifier = f"idx-{index}"
    return f"lt-{identifier}"


def _memory_candidates_from_records(
    records: Sequence[Any],
    recency_window: float,
) -> list[MemoryCandidate]:
    now = datetime.now(timezone.utc)
    candidates: list[MemoryCandidate] = []
    base_window = max(recency_window, 1.0)
    record_count = max(len(records), 1)
    for index, record in enumerate(records):
        candidate_key = _memory_key(record, index)
        created_at = getattr(record, "created_at", None)
        if isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (now - created_at).total_seconds())
        else:
            age_seconds = index * (base_window / record_count)
        recency = max(0.0, min(1.0, 1.0 - age_seconds / base_window))
        strength = float(getattr(record, "strength", 0.5) or 0.0)
        salience = max(0.0, min(1.0, strength))
        attributes = getattr(record, "attributes", {}) or {}
        safety_value = attributes.get("safety", 0.8)
        try:
            safety = float(safety_value)
        except (TypeError, ValueError):
            safety = 0.8
        safety = max(0.0, min(1.0, safety))
        mood = (getattr(record, "mood", "") or "").lower()
        if mood in {"stressed", "anxious", "worried", "afraid"}:
            safety = min(safety, 0.45)
        endocrine = attributes.get("endocrine") if isinstance(attributes, dict) else {}
        normalized = endocrine.get("normalized") if isinstance(endocrine, dict) else {}
        bands = endocrine.get("bands") if isinstance(endocrine, dict) else {}
        spikes: dict[str, float] = {}
        raw_tags = attributes.get("tags") or ()
        if isinstance(raw_tags, str):
            tag_iterable = (raw_tags,)
        else:
            tag_iterable = tuple(raw_tags)
        tag_list = list(tag_iterable)
        if mood:
            tag_list.append(mood)
        for hormone, band in (bands or {}).items():
            if band and band != "steady":
                tag_list.append(f"spike:{hormone}:{band}")
        for hormone, value in (normalized or {}).items():
            try:
                spikes[hormone] = float(value)
            except (TypeError, ValueError):
                continue
        tags = tuple(dict.fromkeys(tag for tag in tag_list if tag))
        candidates.append(
            MemoryCandidate(
                key=candidate_key,
                recency=recency,
                salience=salience,
                safety=safety,
                tags=frozenset(tag for tag in tags if tag),
                spikes=spikes,
            )
        )
    return candidates


def memory_preview(
    traits: TraitSnapshot | None,
    records: Sequence[Any],
    *,
    state_engine: StateEngine,
    recency_window_seconds: float,
    affect_memory_preview_enabled: bool,
    scored_candidates: Sequence[tuple[MemoryCandidate, float]] | None = None,
) -> list[dict[str, Any]]:
    if not (affect_memory_preview_enabled and traits and records):
        return []
    scored = list(scored_candidates or [])
    if not scored:
        endocrine_snapshot = state_engine.endocrine_snapshot()
        hormone_bands = {}
        if isinstance(endocrine_snapshot, dict):
            hormone_bands = endocrine_snapshot.get("bands", {}) or {}
        candidates = _memory_candidates_from_records(records, recency_window_seconds)
        if not candidates:
            return []
        scored = score_memories(traits, candidates, hormone_bands=hormone_bands)
    preview: list[dict[str, Any]] = []
    for candidate, score in scored[: min(5, len(scored))]:
        preview.append(
            {
                "key": candidate.key,
                "score": round(score, 4),
                "tags": sorted(candidate.tags),
            }
        )
    return preview


def _score_memory_candidates(
    *,
    traits: TraitSnapshot | None,
    records: Sequence[Any],
    state_engine: StateEngine,
    recency_window_seconds: float,
) -> tuple[list[MemoryCandidate], list[tuple[MemoryCandidate, float]], TraitSnapshot, dict[str, str]]:
    if not records:
        zero_traits = traits or TraitSnapshot(steadiness=0.0, curiosity=0.0, warmth=0.0, tension=0.0)
        return [], [], zero_traits, {}
    candidates = _memory_candidates_from_records(records, recency_window_seconds)
    if not candidates:
        zero_traits = traits or TraitSnapshot(steadiness=0.0, curiosity=0.0, warmth=0.0, tension=0.0)
        return [], [], zero_traits, {}
    endocrine_snapshot = state_engine.endocrine_snapshot()
    hormone_bands = {}
    if isinstance(endocrine_snapshot, dict):
        hormone_bands = endocrine_snapshot.get("bands", {}) or {}
    scoring_traits = traits or TraitSnapshot(steadiness=0.0, curiosity=0.0, warmth=0.0, tension=0.0)
    scored = score_memories(scoring_traits, candidates, hormone_bands=hormone_bands)
    return candidates, scored, scoring_traits, hormone_bands


def _memory_spotlight(
    *,
    candidates: Sequence[MemoryCandidate],
    scored_candidates: Sequence[tuple[MemoryCandidate, float]],
    traits: TraitSnapshot,
    hormone_bands: Mapping[str, str],
    long_term_pairs: list[tuple[str, Any]],
    limit: int = 4,
) -> list[dict[str, Any]]:
    if not candidates or not scored_candidates:
        return []
    score_lookup = {candidate.key: score for candidate, score in scored_candidates}
    selection = select_memories(
        traits,
        candidates,
        limit=min(limit, len(candidates)),
        hormone_bands=hormone_bands,
    )
    payload_lookup = {key: payload for key, payload in long_term_pairs}
    spotlight: list[dict[str, Any]] = []
    for candidate in selection:
        payload = payload_lookup.get(candidate.key)
        if payload is None:
            continue
        spotlight.append(
            {
                "key": candidate.key,
                "score": round(score_lookup.get(candidate.key, 0.0), 4),
                "record": payload,
            }
        )
        if len(spotlight) >= limit:
            break
    return spotlight


def build_chat_context(
    *,
    state_engine: StateEngine,
    local_llama_engine: LocalLlamaEngine | None,
    persona_snapshot: dict[str, Any],
    affect_context: dict[str, Any] | None,
    affect_memory_preview_enabled: bool,
    recency_window_seconds: float,
    self_narration_note: str | None,
    long_term_limit: int = 5,
) -> dict[str, Any]:
    hormones = state_engine.hormone_system.get_state()
    memory_manager = state_engine.memory_manager
    summary = memory_manager.summarize_recent()
    working = memory_manager.working_snapshot()
    records = list(memory_manager.recent_long_term(limit=long_term_limit))
    long_term_pairs: list[tuple[str, Any]] = []
    for index, record in enumerate(records):
        key = _memory_key(record, index)
        if hasattr(record, "model_dump"):
            payload = record.model_dump()
        elif hasattr(record, "__dict__"):
            payload = dict(record.__dict__)
        else:  # pragma: no cover - defensive fallback
            payload = record
        long_term_pairs.append((key, payload))
    llama_metrics: dict[str, Any] | None = None
    if local_llama_engine is not None:
        try:
            diagnostics = local_llama_engine.diagnostics()
        except Exception:  # pragma: no cover - defensive guard
            diagnostics = None
        if diagnostics:
            llama_metrics = diagnostics

    persona = dict(persona_snapshot)
    if self_narration_note:
        persona["internal_note"] = self_narration_note

    traits = state_engine.trait_snapshot()
    candidates, scored_candidates, scoring_traits, hormone_bands = _score_memory_candidates(
        traits=traits,
        records=records,
        state_engine=state_engine,
        recency_window_seconds=recency_window_seconds,
    )
    spotlight_entries = _memory_spotlight(
        candidates=candidates,
        scored_candidates=scored_candidates,
        traits=scoring_traits,
        hormone_bands=hormone_bands,
        long_term_pairs=long_term_pairs,
    )
    if spotlight_entries:
        ranking = {entry["key"]: idx for idx, entry in enumerate(spotlight_entries)}
        long_term_pairs.sort(key=lambda pair: (ranking.get(pair[0], len(ranking)), pair[0]))

    internal_reflections = memory_manager.recent_internal_reflections(limit=3)
    memory_block: dict[str, Any] = {
        "summary": summary,
        "working": working,
        "long_term": [payload for _, payload in long_term_pairs],
    }
    if internal_reflections:
        memory_block["internal_reflections"] = internal_reflections
    if spotlight_entries:
        memory_block["spotlight"] = spotlight_entries
    if affect_context:
        memory_block["affect_tags"] = affect_context.get("tags", [])
        preview = memory_preview(
            traits,
            records,
            state_engine=state_engine,
            recency_window_seconds=recency_window_seconds,
            affect_memory_preview_enabled=affect_memory_preview_enabled,
            scored_candidates=scored_candidates,
        )
        if preview:
            ranking = {entry["key"]: rank for rank, entry in enumerate(preview)}
            long_term_pairs.sort(key=lambda pair: (ranking.get(pair[0], len(ranking)), pair[0]))
            memory_block["long_term"] = [payload for _, payload in long_term_pairs]
            memory_block["affect_preview"] = preview
    if self_narration_note:
        memory_block["self_note"] = self_narration_note

    context: dict[str, Any] = {
        "mood": state_engine.state.get("mood", "neutral"),
        "hormones": dict(hormones),
        "memory": memory_block,
        "timestamp": state_engine.state.get("timestamp"),
        "llama_metrics": llama_metrics,
        "persona": persona,
    }
    if self_narration_note:
        context["self_note"] = self_narration_note
    if internal_reflections:
        context["inner_reflections"] = list(internal_reflections)
    if affect_context:
        context["affect"] = affect_context
    return context


__all__ = ["build_chat_context", "memory_preview"]
