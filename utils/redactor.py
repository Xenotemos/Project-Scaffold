"""Helpers for stripping internal instrumentation markers from text."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_MARKER_PATTERN = re.compile(r"<<(?:HORMONE|TRAIT|AFFECT):[^>]+>>")
_DEBUG_PATTERN = re.compile(r"\[\[(?:DEBUG|MONITOR):[^\]]+\]\]")


def strip_internal_markers(text: str) -> str:
    """Remove hormone/trait debugging tokens from the supplied text."""
    cleaned = _MARKER_PATTERN.sub("", text)
    cleaned = _DEBUG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def strip_markers_from_messages(messages: Sequence[str]) -> list[str]:
    """Return a new list with markers stripped from each message."""
    return [strip_internal_markers(message) for message in messages]


def contains_markers(text: str) -> bool:
    """Detect whether the text still contains internal markers."""
    return bool(_MARKER_PATTERN.search(text) or _DEBUG_PATTERN.search(text))


def sanitize_blocks(blocks: Iterable[str]) -> tuple[str, ...]:
    """Strip markers from a series of blocks and drop empties."""
    return tuple(block for block in (strip_internal_markers(block) for block in blocks) if block)


__all__ = [
    "strip_internal_markers",
    "strip_markers_from_messages",
    "contains_markers",
    "sanitize_blocks",
]
