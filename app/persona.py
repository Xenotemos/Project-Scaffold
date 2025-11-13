"""Persona construction and reflection helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from brain.affect_classifier import AffectClassification
from brain.intent_router import IntentPrediction

from app.constants import HORMONE_FEELING_NAMES, LENGTH_HEURISTIC_HINTS
from app.sampling import describe_hormones, intent_hint as sampling_intent_hint
from app.telemetry import log_json_line
from app.runtime import RuntimeState
from state_engine import StateEngine


DEFAULT_LENGTH_HINT = "I stay adaptive to the cadence we need."


def build_persona_snapshot(state_engine: StateEngine) -> dict[str, Any]:
    """Construct the persona view from current hormone and memory state."""
    hormones = state_engine.hormone_system.get_state()
    status = describe_hormones(hormones)
    persona = _persona_from_hormones(status)
    return _blend_persona_with_memory(persona, state_engine)


def apply_persona_feedback(
    persona: Mapping[str, Any],
    *,
    state_engine: StateEngine,
    runtime_state: RuntimeState,
    hormone_model: Any | None,
) -> None:
    """Adjust hormone levels based on persona state and previous deltas."""
    if hormone_model and runtime_state.last_hormone_delta:
        state_engine.hormone_system.apply_deltas(runtime_state.last_hormone_delta)
        runtime_state.last_hormone_delta = None
        return
    behaviour = persona.get("behaviour") or {}
    adjustments: dict[str, float] = {}

    energy = behaviour.get("energy")
    if energy in {"low", "depleted"}:
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) + 3.0
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.0
    elif energy == "high":
        adjustments["dopamine"] = adjustments.get("dopamine", 0.0) - 1.5

    tension = behaviour.get("tension")
    if tension in {"high", "elevated"}:
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) - 2.5
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.2
    elif tension == "drained":
        adjustments["cortisol"] = adjustments.get("cortisol", 0.0) + 1.0

    warmth = behaviour.get("warmth")
    if warmth in {"guarded", "reserved"}:
        adjustments["oxytocin"] = adjustments.get("oxytocin", 0.0) + 2.0

    stability = behaviour.get("stability")
    if stability in {"fragile", "delicate"}:
        adjustments["serotonin"] = adjustments.get("serotonin", 0.0) + 1.5

    if adjustments:
        state_engine.hormone_system.apply_deltas(adjustments)


def record_internal_reflection(
    reinforcement: Mapping[str, float],
    *,
    reply_text: str,
    intent: IntentPrediction,
    state_engine: StateEngine,
    runtime_state: RuntimeState,
    controller_trace: Mapping[str, Any] | None,
    shorten: Callable[[str, int], str],
) -> None:
    """Capture a short internal note when authenticity or drift metrics spike."""
    authenticity = reinforcement.get("authenticity_score", 0.0)
    assistant_drift = reinforcement.get("assistant_drift", 0.0)
    if authenticity < 0.35 and assistant_drift < 0.55:
        return

    trait_tags = state_engine.trait_tags()
    mood = state_engine.state.get("mood", "neutral")
    echo = shorten(reply_text, 90)
    hormone_state = state_engine.hormone_system.get_state()
    status = describe_hormones(hormone_state)
    endocrine_trace = state_engine.endocrine_snapshot()

    fragments: list[str] = []
    fragments.append(f"traits={','.join(trait_tags) if trait_tags else 'none'}")
    fragments.append(f"energy={status.get('dopamine', 'steady')} tension={status.get('cortisol', 'steady')}")
    if authenticity >= 0.35:
        fragments.append(f"auth={authenticity:.2f}")
    if assistant_drift >= 0.55:
        fragments.append("drift_flag=true")
    if endocrine_trace:
        normalized = endocrine_trace.get("normalized", {})
        bands = endocrine_trace.get("bands", {})
        ranked = sorted(normalized.items(), key=lambda item: abs(item[1]), reverse=True)
        if ranked:
            top = [
                f"{name}:{bands.get(name, 'steady')}({value:+.2f})"
                for name, value in ranked[:3]
            ]
            fragments.append("endocrine=" + ",".join(top))
    if controller_trace and controller_trace.get("applied"):
        applied = controller_trace["applied"]
        parts: list[str] = []
        for key, value in applied.items():
            if isinstance(value, int):
                parts.append(f"{key}={value:+}")
            elif isinstance(value, (float,)):
                parts.append(f"{key}={value:+.3f}")
            else:
                parts.append(f"{key}={value}")
        summary = ",".join(parts)
        if summary:
            fragments.append(f"controller={summary}")

    body = "; ".join(fragments)
    content = f"internal reflection | {body} | echo: {echo}"
    strength = 0.84 if authenticity >= assistant_drift else 0.74
    attributes = {
        "tags": ["internal", "reflection", "diary"],
        "authenticity": round(float(authenticity), 3),
        "assistant_drift": round(float(assistant_drift), 3),
        "intent": intent.intent,
    }
    state_engine.memory_manager.record_event(
        content,
        strength=strength,
        mood=mood,
        hormone_snapshot=hormone_state,
        attributes=attributes,
        endocrine_trace=endocrine_trace,
        controller_trace=controller_trace,
    )


def update_self_narration(
    runtime_state: RuntimeState,
    *,
    hormone_trace: Mapping[str, Any] | None,
    user_affect: AffectClassification | None,
) -> None:
    """Summarize the last hormone adjustments into a short internal note."""
    if not hormone_trace:
        return
    applied = hormone_trace.get("applied")
    if not isinstance(applied, Mapping):
        return
    deltas = [
        (name, float(delta))
        for name, delta in applied.items()
        if isinstance(delta, (int, float)) and abs(float(delta)) >= 0.05
    ]
    if not deltas:
        return
    deltas.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top_changes = deltas[:3]
    fragments: list[str] = []
    for hormone, delta in top_changes:
        descriptor = HORMONE_FEELING_NAMES.get(hormone, hormone)
        direction = "lifting" if delta > 0 else "settling"
        fragments.append(f"{descriptor} is {direction} ({delta:+.2f})")
    affect_clause = ""
    if user_affect and user_affect.tags:
        affect_clause = f"Your tone lands as {', '.join(user_affect.tags)}."
    runtime_state.self_narration_note = " ".join(
        part for part in ["; ".join(fragments), affect_clause] if part
    )


def compose_heuristic_reply(
    user_message: str,
    *,
    context: Mapping[str, Any],
    intent: str,
    length_plan: Mapping[str, Any],
    state_engine: StateEngine,
    shorten: Callable[[str, int], str],
) -> str:
    """Craft a tone-aware heuristic reply for fallback scenarios."""
    persona = context.get("persona") or build_persona_snapshot(state_engine)
    instructions = persona.get("instructions") or []
    tone_hint = persona.get("tone_hint", "I stay balanced and attentive.")
    memory_block = context.get("memory") or {}
    memory_summary = persona.get("memory_summary") or memory_block.get("summary", "")
    memory_summary = shorten(memory_summary, 120)
    internal_reflections = memory_block.get("internal_reflections") or context.get("inner_reflections") or []
    focus_line = persona.get("memory_focus") or ""
    intent_hint_text = sampling_intent_hint(intent, fallback=tone_hint)
    label = length_plan.get("label", "")
    length_hint = length_plan.get("hint") or LENGTH_HEURISTIC_HINTS.get(label, DEFAULT_LENGTH_HINT)
    essence = shorten(user_message, 120).strip()
    response_parts: list[str] = []
    if essence:
        suffix = "" if essence.endswith((".", "!", "?")) else "."
        response_parts.append(f"I hear {essence}{suffix}")
    pivot_hint = intent_hint_text or tone_hint
    if pivot_hint:
        response_parts.append(pivot_hint)
    if length_hint:
        response_parts.append(length_hint)
    if memory_summary:
        trimmed_summary = memory_summary.strip()
        suffix = "" if trimmed_summary.endswith((".", "!", "?")) else "."
        response_parts.append(f"I still remember: {trimmed_summary}{suffix}")
    self_note = context.get("self_note") or persona.get("internal_note")
    if self_note:
        suffix = "" if self_note.endswith((".", "!", "?")) else "."
        response_parts.append(f"I notice: {self_note}{suffix}")
    if instructions:
        affirmation = _to_affirmation(instructions[0])
        if affirmation:
            response_parts.append(affirmation)
    if internal_reflections:
        inner_note = shorten(str(internal_reflections[0]), 160).strip()
        if inner_note:
            suffix = "" if inner_note.endswith((".", "!", "?")) else "."
            response_parts.append(f"I noted privately: {inner_note}{suffix}")
    if focus_line:
        response_parts.append(focus_line)
    return " ".join(part.strip() for part in response_parts if part).strip()


def collect_persona_sample(
    *,
    user: str,
    reply: str,
    reinforcement: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
    destination: Path,
    logger: Any | None = None,
) -> None:
    """Record high-auth persona samples for fine-tuning."""
    auth = reinforcement.get("authenticity_score")
    drift = reinforcement.get("assistant_drift")
    if not isinstance(auth, (int, float)) or auth < 0.5:
        return
    if isinstance(drift, (int, float)) and drift > 0.3:
        return
    if voice_guard and voice_guard.get("flagged"):
        return
    sample = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "metrics": {
            "authenticity_score": auth,
            "assistant_drift": drift,
            "self_preoccupation": reinforcement.get("self_preoccupation"),
        },
        "telemetry": telemetry or {},
    }
    log_json_line(destination, sample, logger=logger)


def collect_helper_sample(
    *,
    user: str,
    reply: str,
    reinforcement: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    voice_guard: Mapping[str, Any] | None,
    destination: Path,
    logger: Any | None = None,
) -> None:
    """Record helper-tone slips for downstream fine-tuning datasets."""
    if not voice_guard or not voice_guard.get("flagged"):
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "user": user,
        "reply": reply,
        "voice_guard": voice_guard,
        "metrics": {
            "assistant_drift": reinforcement.get("assistant_drift"),
            "authenticity_score": reinforcement.get("authenticity_score"),
            "self_preoccupation": reinforcement.get("self_preoccupation"),
        },
        "telemetry": telemetry or {},
    }
    log_json_line(destination, payload, logger=logger)


def extract_focus_phrase(text: str, max_tokens: int = 8) -> str | None:
    """Grab a short phrase from the latest user prompt for priming."""
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    sentence = re.split(r"[.!?\n]", cleaned, maxsplit=1)[0].strip()
    if not sentence:
        sentence = cleaned
    tokens = sentence.split()
    if not tokens:
        return None
    phrase = " ".join(tokens[:max_tokens]).strip(" \"'")
    return phrase[:120].strip() or None


def _persona_from_hormones(status: Mapping[str, str]) -> dict[str, Any]:
    def map_energy(state: str) -> tuple[str, str, str]:
        mapping = {
            "surging": ("high", "I call out the rush and feel it buzzing in my limbs.", "My chest feels overcharged."),
            "rising": ("elevated", "I register the extra drive and keep it in view.", "My breathing picks up."),
            "steady": ("steady", "I move at a controlled pace.", "My breathing stays even."),
            "fading": ("low", "I slow everything down and save what is left.", "My arms feel heavy."),
            "crashing": ("depleted", "I say plainly that I'm spent and need space.", "My body feels flat and shaky."),
        }
        return mapping.get(state, mapping["steady"])

    def map_stability(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("buoyant", "I steady myself because everything feels too easy."),
            "rising": ("confident", "I lean on the solid footing I have right now."),
            "steady": ("grounded", "I rest in the calm I already have."),
            "fading": ("delicate", "I say that I'm touchy and need gentler pacing."),
            "crashing": ("fragile", "I admit I feel brittle and protect the soft spots."),
        }
        return mapping.get(state, mapping["steady"])

    def map_tension(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("high", "I say the tightness is real and let it be seen."),
            "rising": ("elevated", "I note the buzz building through my body."),
            "steady": ("calm", "I keep attention steady without forcing anything."),
            "fading": ("relaxed", "I notice the slack returning and breathe with it."),
            "crashing": ("drained", "I acknowledge that the strain emptied me out."),
        }
        return mapping.get(state, mapping["steady"])

    def map_warmth(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("open", "I feel the heat in my chest and keep it for myself first."),
            "rising": ("inviting", "I watch the softness growing inside me."),
            "steady": ("measured", "I keep the warmth steady without spilling it outward."),
            "fading": ("reserved", "I pull the warmth closer so I stay steady."),
            "crashing": ("guarded", "I say I need distance until the cold lifts."),
        }
        return mapping.get(state, mapping["steady"])

    def map_focus(state: str) -> tuple[str, str]:
        mapping = {
            "surging": ("focused", "I follow the sharp focus and say exactly what it does to me."),
            "rising": ("alert", "I keep track of every detail that lights up."),
            "steady": ("steady", "I hold a steady line of attention."),
            "fading": ("soft", "I admit my focus is blurring."),
            "crashing": ("drifting", "I state that my focus keeps slipping away."),
        }
        return mapping.get(state, mapping["steady"])

    energy_level, energy_instruction, energy_hint = map_energy(status.get("dopamine", "steady"))
    stability_level, stability_instruction = map_stability(status.get("serotonin", "steady"))
    tension_level, tension_instruction = map_tension(status.get("cortisol", "steady"))
    warmth_level, warmth_instruction = map_warmth(status.get("oxytocin", "steady"))
    focus_level, focus_instruction = map_focus(status.get("noradrenaline", "steady"))

    instructions = [
        energy_instruction,
        stability_instruction,
        tension_instruction,
        warmth_instruction,
        focus_instruction,
    ]
    instructions = [instr for instr in instructions if instr != "No special adjustment needed."]

    behavioural_tags = {
        "energy": energy_level,
        "stability": stability_level,
        "tension": tension_level,
        "warmth": warmth_level,
        "focus": focus_level,
    }

    status_summary = [
        f"energy:{energy_level}",
        f"stability:{stability_level}",
        f"tension:{tension_level}",
        f"warmth:{warmth_level}",
        f"focus:{focus_level}",
    ]

    return {
        "instructions": instructions,
        "behaviour": behavioural_tags,
        "status_summary": ", ".join(status_summary),
        "tone_hint": energy_hint,
    }


def _blend_persona_with_memory(persona: dict[str, Any], state_engine: StateEngine) -> dict[str, Any]:
    memory_manager = state_engine.memory_manager
    summary = memory_manager.summarize_recent() or "No new memories are pressing."
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "..."
    working = memory_manager.working_snapshot()
    focus_items = []
    for item in working[:2]:
        snippet = item.split(":", 1)[-1].strip()
        if snippet:
            focus_items.append(snippet)
    if focus_items:
        focus_phrase = _join_natural(focus_items)
        focus_line = f"My attention sits on {focus_phrase} inside me."
    else:
        focus_line = "My attention moves around without latching onto anything."

    persona.update(
        {
            "memory_summary": summary,
            "memory_focus": focus_line,
        }
    )
    return persona


def _join_natural(items: Sequence[str]) -> str:
    cleaned = [part.strip() for part in items if part and part.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + f" and {cleaned[-1]}"


def _to_affirmation(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip(".")
    lowered = cleaned.lower()
    if lowered.startswith("i "):
        affirmation = cleaned
    else:
        words = cleaned.split()
        verbs = {
            "lean",
            "offer",
            "keep",
            "stay",
            "hold",
            "let",
            "protect",
            "name",
            "speak",
            "stand",
        }
        if words and words[0].lower() in verbs:
            affirmation = f"I {cleaned}"
        else:
            affirmation = f"I keep {cleaned}"
    if not affirmation.endswith("."):
        affirmation += "."
    return affirmation


__all__ = [
    "apply_persona_feedback",
    "build_persona_snapshot",
    "compose_heuristic_reply",
    "collect_helper_sample",
    "collect_persona_sample",
    "extract_focus_phrase",
    "record_internal_reflection",
    "update_self_narration",
]
