from __future__ import annotations

import asyncio
from typing import Any

from app.persona import build_persona_snapshot, compose_heuristic_reply
from state_engine import StateEngine


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    shortened = text[: limit - 3].rsplit(" ", 1)[0]
    return f"{shortened}..."


class PersonaHelperTests:
    def setup_method(self) -> None:
        self.engine = StateEngine()
        asyncio.run(
            self.engine.register_event(
                "I logged a short diagnostic experience about warmth",
                strength=0.6,
                stimulus_type="reward",
            )
        )

    def _base_context(self) -> dict[str, Any]:
        snapshot = build_persona_snapshot(self.engine)
        manager = self.engine.memory_manager
        return {
            "persona": snapshot,
            "memory": {
                "summary": manager.summarize_recent(),
                "working": manager.working_snapshot(),
                "internal_reflections": ["I noticed my breathing slow down as I listened."],
                "long_term": [],
            },
            "self_note": "My ribs loosen when I echo their words.",
        }

    def test_compose_heuristic_reply_mentions_user_content(self) -> None:
        context = self._base_context()
        length_plan = {"label": "concise", "hint": "Keep it brief but embodied."}
        reply = compose_heuristic_reply(
            "diagnostic ping about aching shoulders",
            context=context,
            intent="analytical",
            length_plan=length_plan,
            state_engine=self.engine,
            shorten=_shorten,
        )
        assert reply.strip()
        assert "diagnostic ping" in reply.lower()
        assert reply.lower().startswith("i hear ")

    def test_compose_heuristic_reply_blends_memory_summary(self) -> None:
        context = self._base_context()
        context["memory"]["summary"] = "We shared a heavy story yesterday."
        reply = compose_heuristic_reply(
            "tell me what you notice",
            context=context,
            intent="reflective",
            length_plan={"label": "balanced", "hint": "Aim for two sentences."},
            state_engine=self.engine,
            shorten=_shorten,
        )
        assert "i still remember" in reply.lower()
