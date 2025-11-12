"""Heuristic voice guard discouraging helper persona drift."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, List

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s*")
_HELPER_PHRASES = (
    "how can i help",
    "let me know if",
    "i'm here to help",
    "i am here to help",
    "i'm here to assist",
    "i am here to assist",
    "please let me know",
    "i can assist",
    "happy to help",
    "support you",
    "assist you",
    "anything you need",
    "let me support",
    "in response to your request",
    "we can",
    "we should",
    "together we",
    "let's",
    "let us",
    "open to new experiences",
    "let's explore and learn together",
    "i'm always open to new experiences",
    "i am an artificial intelligence",
    "i'm an artificial intelligence",
    "i am an ai",
    "i'm an ai",
    "i am a virtual assistant",
    "i'm a virtual assistant",
    "i was designed to assist",
    "i am designed to assist",
    "my primary function is to assist",
    "i was designed to help",
    "i am designed to help",
)
_REPETITIVE_PHRASES = (
    "let's explore and learn together",
    "open to new experiences and ready for interaction",
    "i'm feeling neutral at the moment",
    "i'm always open to new experiences",
    "how about you? how have you been lately?",
)
_COLLABORATIVE_PATTERN = re.compile(r"\b(let's|let us|we can|we should|together we|we will)\b", re.IGNORECASE)
_INVITATION_PATTERN = re.compile(r"\b(please let me know|let me know if|anything else you need|reach out if|feel free to ask)\b", re.IGNORECASE)



def _normalise(text: str) -> str:
    return (text or "").strip().lower()


@dataclass
class VoiceGuardVerdict:
    score: float = 0.0
    categories: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)
    repeated_sentences: List[str] = field(default_factory=list)
    flagged: bool = False
    severity: str = "clear"

    def to_dict(self) -> dict[str, Any]:
        return {
            "flagged": self.flagged,
            "severity": self.severity,
            "score": round(self.score, 3),
            "categories": list(self.categories),
            "matches": list(dict.fromkeys(self.matches)),
            "repeated_sentences": list(dict.fromkeys(self.repeated_sentences)),
        }


class VoiceGuard:
    def __init__(self, *, penalty_threshold: float = 0.45) -> None:
        self.penalty_threshold = penalty_threshold

    def evaluate(self, reply: str | None) -> VoiceGuardVerdict:
        text = (reply or "").strip()
        if not text:
            return VoiceGuardVerdict()
        normalized = _normalise(text)
        categories: list[str] = []
        matches: list[str] = []
        score = 0.0

        helper_hits = _collect_hits(normalized, _HELPER_PHRASES)
        if helper_hits:
            categories.append("helper_phrasing")
            matches.extend(helper_hits)
            score += 0.5 + 0.1 * (len(helper_hits) - 1)

        collab_hits = _COLLABORATIVE_PATTERN.findall(text)
        if collab_hits:
            categories.append("collaborative_tone")
            matches.extend([hit.lower() for hit in collab_hits])
            score += 0.25 + 0.05 * (len(collab_hits) - 1)

        invitation_hits = _INVITATION_PATTERN.findall(normalized)
        if invitation_hits:
            categories.append("soliciting_more_requests")
            matches.extend([hit.lower() for hit in invitation_hits])
            score += 0.2 + 0.05 * (len(invitation_hits) - 1)

        repetitive_phrase_hits = _collect_hits(normalized, _REPETITIVE_PHRASES)
        if repetitive_phrase_hits:
            categories.append("memorised_phrase")
            matches.extend(repetitive_phrase_hits)
            score += 0.18 + 0.04 * (len(repetitive_phrase_hits) - 1)

        repeated_sentences = _detect_repeated_sentences(text)
        if repeated_sentences:
            categories.append("repetition")
            score += 0.22 + 0.05 * (len(repeated_sentences) - 1)

        score = min(score, 1.0)
        flagged = score >= self.penalty_threshold or bool(repeated_sentences)
        severity = "penalize" if flagged else "clear"

        verdict = VoiceGuardVerdict(
            score=score,
            categories=_dedupe(categories),
            matches=_dedupe(matches),
            repeated_sentences=_dedupe(repeated_sentences),
            flagged=flagged,
            severity=severity,
        )
        return verdict


def _collect_hits(text: str, phrases: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for phrase in phrases:
        if phrase in text:
            hits.append(phrase)
    return hits


def _detect_repeated_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for sentence in sentences:
        key = sentence.lower()
        if len(key) < 12:
            continue
        if key in seen and sentence not in duplicates:
            duplicates.append(sentence)
        else:
            seen[key] = seen.get(key, 0) + 1
    return duplicates


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


HELPER_PHRASES = _HELPER_PHRASES
REPETITIVE_PHRASES = _REPETITIVE_PHRASES

__all__ = [
    "VoiceGuard",
    "VoiceGuardVerdict",
    "HELPER_PHRASES",
    "REPETITIVE_PHRASES",
]
