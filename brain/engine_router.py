"""Routing stubs for future multi-engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from brain.engines import Engine, EngineProfile


@dataclass(frozen=True)
class RoutingContext:
    """Task description used when selecting an engine."""

    task: str
    required_roles: tuple[str, ...] = ()
    preferred_size: Optional[str] = None


@dataclass(frozen=True)
class RoutingDecision:
    """Result returned by the router."""

    engine_name: str
    rationale: str


class EngineRouter:
    """Minimal router that can be expanded once additional engines exist."""

    def __init__(self) -> None:
        self._registry: Dict[str, Engine] = {}

    def register(self, engine: Engine) -> None:
        self._registry[engine.profile.name] = engine

    def registered_profiles(self) -> List[EngineProfile]:
        return [engine.profile for engine in self._registry.values()]

    def route(self, context: RoutingContext) -> RoutingDecision:
        if not self._registry:
            raise LookupError("no engines registered")

        candidates = list(self._filter_by_role(context.required_roles))
        if context.preferred_size:
            sized = [engine for engine in candidates if engine.profile.model_size == context.preferred_size]
            if sized:
                candidates = sized

        chosen = candidates[0] if candidates else next(iter(self._registry.values()))
        rationale = self._build_rationale(context, chosen.profile)
        return RoutingDecision(engine_name=chosen.profile.name, rationale=rationale)

    def _filter_by_role(self, required_roles: Iterable[str]) -> Iterable[Engine]:
        roles = tuple(required_roles)
        if not roles:
            return self._registry.values()
        return (
            engine
            for engine in self._registry.values()
            if all(engine.profile.supports_role(role) for role in roles)
        )

    @staticmethod
    def _build_rationale(context: RoutingContext, profile: EngineProfile) -> str:
        pieces = [f"task={context.task}"]
        if context.required_roles:
            pieces.append(f"roles={','.join(context.required_roles)}")
        pieces.append(f"engine={profile.name}")
        pieces.append(f"size={profile.model_size}")
        return "; ".join(pieces)


__all__ = ["EngineRouter", "RoutingContext", "RoutingDecision"]
