"""Shared interfaces and metadata for language model engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class EngineProfile:
    """Describes the capabilities of a particular engine instance."""

    name: str
    model_size: str
    parameters_billion: float
    roles: tuple[str, ...]
    supports_streaming: bool = True
    max_context_tokens: int | None = None
    notes: str | None = None

    def supports_role(self, role: str) -> bool:
        return role in self.roles


@runtime_checkable
class Engine(Protocol):
    """Protocol implemented by conversational engines."""

    profile: EngineProfile

    async def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        settings: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """Produce a model response for the supplied messages."""
        raise NotImplementedError


__all__ = ["EngineProfile", "Engine"]
