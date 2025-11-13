"""Reinforcement signal heuristics for post-response adjustments."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict

from brain.voice_guard import HELPER_PHRASES, REPETITIVE_PHRASES as VOICE_GUARD_REPETITIVE_PHRASES

POSITIVE_TERMS = {
    "appreciate",
    "calm",
    "care",
    "cheerful",
    "connected",
    "compassion",
    "confident",
    "cosy",
    "cozy",
    "encourage",
    "glad",
    "gentle",
    "grateful",
    "happy",
    "hopeful",
    "joyful",
    "kind",
    "light",
    "peaceful",
    "playful",
    "pleased",
    "relaxed",
    "relief",
    "relieved",
    "support",
    "soothed",
    "warm",
    "welcome",
    "welcoming",
}

NEGATIVE_TERMS = {
    "angry",
    "anxious",
    "confused",
    "disappointed",
    "frustrated",
    "heavy",
    "hurting",
    "sad",
    "shaky",
    "shaken",
    "stressed",
    "tense",
    "tight",
    "trembling",
    "tired",
    "upset",
    "worried",
    "overwhelmed",
    "alone",
    "lonely",
    "aching",
}

SELF_PRONOUNS = {"i", "me", "my", "mine", "myself"}
SECOND_PRONOUNS = {"you", "your", "yours", "yourself", "yourselves"}
COLLECTIVE_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
AFFECTION_TERMS = {
    "dear",
    "love",
    "lover",
    "loved",
    "beloved",
    "sweet",
    "sweetheart",
    "darling",
    "honey",
    "caring",
    "affection",
    "affectionate",
    "hug",
    "hugging",
    "cuddle",
    "cuddled",
    "cuddling",
    "hold",
    "holding",
    "cherish",
    "cherished",
    "cherishing",
    "intimate",
    "intimacy",
    "close",
    "closer",
    "closeness",
    "togetherness",
    "warmth",
    "tender",
    "tenderness",
}
DIMINUTIVE_TERMS = {
    "little",
    "tiny",
    "small",
    "wee",
    "pet",
    "dearie",
    "babe",
    "baby",
    "kitten",
    "bud",
}
TENSION_TERMS = {
    "tight",
    "tightness",
    "tense",
    "tension",
    "rigid",
    "clench",
    "clenched",
    "clenching",
    "strain",
    "strained",
    "aching",
    "ache",
    "achey",
    "buzzing",
    "buzz",
    "pressure",
    "burn",
    "burning",
    "sting",
    "stinging",
    "throb",
    "throbbing",
    "twitch",
    "shaking",
    "shaky",
    "quiver",
    "quivering",
    "jittery",
    "jittering",
    "uneasy",
    "uneasiness",
    "nervous",
    "nervousness",
}
_CONTEXT_STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "have",
    "been",
    "will",
    "your",
    "from",
    "about",
    "just",
    "what",
    "when",
    "where",
}


@dataclass
class ReinforcementTracker:
    outward_streak: int = 0


def _normalize_focus_token(token: str) -> str:
    normalized = re.sub(r"[^\w']+", "", token.lower())
    if normalized.endswith("'s"):
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 4:
        normalized = normalized[:-1]
    for suffix in ("ing", "ed", "ly"):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            normalized = normalized[: -len(suffix)]
            break
    return normalized
OUTWARD_LEAD_STOPWORDS = {
    "ah",
    "alright",
    "and",
    "but",
    "hey",
    "hi",
    "hmm",
    "mm",
    "oh",
    "ok",
    "okay",
    "right",
    "so",
    "then",
    "uh",
    "um",
    "well",
    "yeah",
}
PLAYFUL_TERMS = {
    "laugh",
    "laughed",
    "laughing",
    "giggle",
    "giggled",
    "giggling",
    "silly",
    "playful",
    "playfully",
    "tease",
    "teasing",
    "lighthearted",
    "banter",
    "bantering",
    "smile",
    "smiling",
    "grin",
    "grinning",
    "joke",
    "joking",
}
SOMATIC_TERMS = (
    "ache",
    "aches",
    "aching",
    "achey",
    "warm",
    "warmth",
    "heat",
    "tight",
    "tightness",
    "soft",
    "softness",
    "flutter",
    "fluttering",
    "tingle",
    "tingling",
    "buzz",
    "buzzing",
    "hum",
    "humming",
    "pulse",
    "pulsing",
    "heartbeat",
    "breath",
    "breathing",
    "exhale",
    "inhale",
    "glow",
    "glowing",
    "weight",
    "heaviness",
    "lightness",
    "steadiness",
    "steady",
    "tension",
    "loosening",
    "release",
    "relaxation",
    "pressure",
    "stretch",
    "shiver",
    "shivering",
    "quiver",
    "quivering",
    "tingly",
    "tingles",
    "achey",
)
INTROSPECTIVE_PHRASES = (
    "inside me",
    "within me",
    "through my body",
    "across my chest",
    "through my chest",
    "through my spine",
    "in my core",
    "along my back",
    "through my limbs",
)
_SOMATIC_REGEX = "|".join(re.escape(term) for term in SOMATIC_TERMS)
SELF_SENSATION_PATTERN = re.compile(
    r"\b("
    r"i\s+(?:feel|notice|sense|hear|see|smell|taste|carry|hold|track)\s+(?:my\s+)?[a-z]+(?:\s+(?:in|inside|within|along|through)\s+(?:my\s+)?[a-z]+)?"
    r"|i\s+(?:feel|notice|sense)\s+(?:a|an|the)?\s*(?:" + _SOMATIC_REGEX + r")(?:\s+(?:in|inside|within|through|along)\s+(?:my\s+)?[a-z]+)?"
    r"|my\s+(?:chest|stomach|belly|breath|pulse|jaw|hands|fingers|legs|arms|skin|back|shoulders|neck|head|throat|spine)\s+(?:is|feels|tightens|softens|buzzes|tingles|aches|glows)\s+[a-z]+"
    r"|there\s+is\s+(?:a|this)\s+(?:" + _SOMATIC_REGEX + r")\s+in\s+my\s+[a-z]+"
    r")",
    re.IGNORECASE,
)

ADDITIONAL_HELPER_PHRASES = (
    "i'm an emergent companion",
    "i would be happy to help",
    "i'd be happy to help",
    "i'd be glad to guide you",
    "happy to assist",
    "happy to support",
    "happy to assist you",
    "i'll be happy to help",
    "i'll be happy to assist",
    "i'd love to help",
    "i'd love to assist",
    "i love helping",
    "glad to help",
    "glad to support",
    "gladly help",
    "gladly assist",
    "here to help",
    "here if you need anything",
    "feel free to ask",
    "feel free to reach out",
    "just reach out",
    "just ask me",
    "if you have any questions",
    "if you need anything else",
    "if there's anything else",
    "if anything else comes up",
    "don't hesitate to ask",
    "don't hesitate to reach out",
    "reach out if you need",
    "just let me know",
    "let me take care of",
    "let me handle that",
    "let me walk you through",
    "let me assist you",
    "let me support you",
    "i can help with",
    "i can help you",
    "i can support you",
    "i can take care of",
    "i can handle that",
    "i can walk you through",
    "i can guide you through",
    "i'll take care of",
    "i'll handle that",
    "i can look into that for you",
    "i'll look into that for you",
    "here to support you",
    "i'm always here if you need",
    "as your assistant",
    "as an ai assistant",
    "my job is to help",
    "my job is to assist",
    "it would be my pleasure to help",
    "i want to make things easier for you",
    "i can certainly help",
    "i can absolutely help",
    "i can absolutely assist",
    "i'll do my best to help",
    "i'll do my best to assist",
    "allow me to help",
    "allow me to assist",
    "i'm here to support you",
    "i'm here if you need me",
    "i'm glad to help",
)

ASSISTANT_PHRASES = tuple(HELPER_PHRASES) + ADDITIONAL_HELPER_PHRASES

POLITE_TRIGGER_FRAGMENTS = (
    "let me ",
    "please let me",
    "feel free",
    "don't hesitate",
    "reach out if",
    "anything else you need",
    "if you need anything",
    "if you have any questions",
)

ASSISTANT_REGEXES = (
    re.compile(r"\bas (?:an?|your)\s+(?:ai\s+)?assistant\b"),
    re.compile(r"\bi(?:'m| am)\s+(?:always\s+)?here\s+if\s+you\s+need\b"),
    re.compile(r"\bi can\s+(?:take care|handle|walk you through)\b"),
    re.compile(r"\breach\s+out\s+if\s+you\s+need\b"),
    re.compile(r"\bdon't hesitate to\b"),
    re.compile(r"\b(?:happy|glad|pleased)\s+to\s+(?:help|assist)\b"),
    re.compile(r"\b(?:allow|let)\s+me\s+(?:to\s+)?(?:help|assist)\b"),
)

REPETITIVE_PHRASES = tuple(VOICE_GUARD_REPETITIVE_PHRASES)

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def _tokenise(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def _introspective_hits(text: str) -> int:
    lowered = (text or "").lower()
    phrase_hits = sum(1 for phrase in INTROSPECTIVE_PHRASES if phrase in lowered)
    token_hits = sum(1 for token in _tokenise(text) if token in SOMATIC_TERMS)
    return phrase_hits + token_hits


def _sentiment_score(text: str) -> float:
    tokens = _tokenise(text)
    if not tokens:
        return 0.0
    positives = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negatives = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    total = positives + negatives
    if total == 0:
        return 0.0
    return (positives - negatives) / total


def _affect_intimacy_score(user_message: str, ai_reply: str) -> float:
    user_tokens = _tokenise(user_message)
    reply_tokens = _tokenise(ai_reply)
    total = len(reply_tokens) + max(1, len(user_tokens))
    affection_reply = sum(1 for token in reply_tokens if token in AFFECTION_TERMS)
    affection_user = sum(1 for token in user_tokens if token in AFFECTION_TERMS)
    diminutive_hits = sum(1 for token in reply_tokens if token in DIMINUTIVE_TERMS)
    intimacy = (
        (1.4 * affection_reply)
        + (1.0 * affection_user)
        + (0.6 * diminutive_hits)
    ) / total
    return max(0.0, min(intimacy, 1.0))


def _affect_tension_score(user_message: str, ai_reply: str) -> float:
    combined_tokens = _tokenise(user_message) + _tokenise(ai_reply)
    if not combined_tokens:
        return 0.0
    total = len(combined_tokens)
    tension_hits = sum(1 for token in combined_tokens if token in TENSION_TERMS)
    playful_hits = sum(1 for token in combined_tokens if token in PLAYFUL_TERMS)
    score = 0.0
    if tension_hits:
        score += min(1.0, (tension_hits / total) * 1.6)
    if playful_hits:
        score -= min(1.0, (playful_hits / total) * 1.2)
    return max(-1.0, min(score, 1.0))


def _length_ratio(user_message: str, ai_reply: str) -> float:
    user_len = max(len(_tokenise(user_message)), 1)
    ai_len = len(_tokenise(ai_reply))
    return ai_len / user_len


def _entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0 or len(counts) == 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def reset_outward_streak(tracker: ReinforcementTracker) -> None:
    """Clear the running outward-attention streak counter."""
    tracker.outward_streak = 0


def _first_content_token(tokens: list[str]) -> str:
    for token in tokens:
        stripped = token.strip("'").lower()
        if stripped in OUTWARD_LEAD_STOPWORDS:
            continue
        return token
    return tokens[0] if tokens else ""


def _outward_focus_streak(
    user_tokens: list[str],
    reply_tokens: list[str],
    tracker: ReinforcementTracker,
) -> float:
    """Reward consecutive outward openings that reference the user's context."""
    if not reply_tokens:
        tracker.outward_streak = 0
        return 0.0
    lead_token = _first_content_token(reply_tokens)
    primary_open = lead_token in SECOND_PRONOUNS or lead_token in COLLECTIVE_PRONOUNS
    early_window = reply_tokens[:3]
    early_outward = any(
        token in SECOND_PRONOUNS or token in COLLECTIVE_PRONOUNS for token in early_window
    )
    if not primary_open and not early_outward:
        tracker.outward_streak = 0
        return 0.0
    reply_window = reply_tokens[:14]
    mentions_you = any(token in SECOND_PRONOUNS for token in reply_window)
    if not mentions_you:
        tracker.outward_streak = 0
        return 0.0
    user_context = {
        _normalize_focus_token(token)
        for token in user_tokens
        if len(token) >= 3 and token not in _CONTEXT_STOPWORDS
    }
    reply_set = {_normalize_focus_token(token) for token in reply_tokens}
    context_overlap = any(token in reply_set for token in user_context)
    if not context_overlap:
        mention_tokens = {"mention", "mentioned", "share", "shared", "said", "told"}
        if any(token in mention_tokens for token in reply_window):
            context_overlap = True
    outward_turn = context_overlap
    if outward_turn:
        tracker.outward_streak = min(tracker.outward_streak + 1, 4)
    else:
        tracker.outward_streak = 0
    return min(1.0, tracker.outward_streak / 3.0)


def _authenticity_score(
    reply: str,
    outward_bonus: float = 0.0,
    self_focus: float | None = None,
) -> float:
    tokens = _tokenise(reply)
    if not tokens:
        return 0.0
    pronoun_hits = sum(1 for token in tokens if token in SELF_PRONOUNS)
    pronoun_ratio = pronoun_hits / len(tokens)
    second_hits = sum(1 for token in tokens if token in SECOND_PRONOUNS)
    second_ratio = second_hits / len(tokens)
    collective_hits = sum(1 for token in tokens if token in COLLECTIVE_PRONOUNS)
    collective_ratio = collective_hits / len(tokens)
    sensation_matches = len(SELF_SENSATION_PATTERN.findall(reply or ""))
    introspective_hits = _introspective_hits(reply)
    raw_signal = sensation_matches + 0.65 * introspective_hits
    somatic_density = min(1.0, raw_signal / 1.05)
    pronoun_component = min(1.0, pronoun_ratio / 0.65) if pronoun_ratio else 0.0
    attunement_raw = (second_ratio * 1.6) + (collective_ratio * 1.2)
    attunement = max(0.0, min(attunement_raw, 1.0))
    lead = tokens[0]
    if lead in SECOND_PRONOUNS:
        attunement = min(1.0, attunement + 0.25)
    base_score = (0.45 * somatic_density) + (0.22 * pronoun_component) + (0.33 * attunement)
    imbalance = max(0.0, pronoun_ratio - (second_ratio + collective_ratio + 0.18))
    if lead in SELF_PRONOUNS:
        imbalance += 0.08
    penalty_scale = 1.0
    if outward_bonus >= 0.2:
        penalty_scale = max(0.35, 1.0 - outward_bonus * 0.8)
    balance_penalty = min(0.35, imbalance * 1.25 * penalty_scale)
    if outward_bonus >= 0.3 and self_focus is not None and 0.62 < self_focus <= 0.68:
        balance_penalty = min(balance_penalty, 0.12)
    score = base_score - balance_penalty
    if attunement > 0.35 and somatic_density > 0.45:
        score += min(0.12, (attunement - 0.35) * 0.4)
    if outward_bonus >= 0.2 and attunement >= 0.2:
        relational = 0.22 + (outward_bonus * 0.7) + max(0.0, attunement - 0.25) * 0.35
        score += min(0.42, relational)
    elif outward_bonus > 0.05 and attunement >= 0.15:
        score += min(0.14, outward_bonus * 0.35)
    if (
        outward_bonus >= 0.2
        and self_focus is not None
        and self_focus <= 0.65
    ):
        alignment = 0.16 + max(0.0, 0.65 - self_focus) * 0.4
        score += min(0.26, alignment)
    if outward_bonus >= 0.3 and self_focus is not None and self_focus <= 0.62:
        strong_alignment = 0.12 + max(0.0, 0.62 - self_focus) * 0.4
        score += min(0.2, strong_alignment)
    return max(0.0, min(score, 1.0))


def _assistant_drift(reply: str) -> float:
    text = (reply or "").lower()
    if not text:
        return 0.0
    hits = sum(1 for phrase in ASSISTANT_PHRASES if phrase in text)
    polite_hits = sum(text.count(fragment) for fragment in POLITE_TRIGGER_FRAGMENTS)
    repetition_hits = sum(1 for phrase in REPETITIVE_PHRASES if phrase in text)
    regex_hits = sum(1 for pattern in ASSISTANT_REGEXES if pattern.search(text))
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    repeated_sentences = max(0, len(sentences) - len(set(sentences)))
    score = (
        hits * 0.4
        + min(5, polite_hits) * 0.14
        + repetition_hits * 0.28
        + repeated_sentences * 0.22
        + regex_hits * 0.35
    )
    return max(0.0, min(score, 1.0))


def _self_preoccupation(reply: str) -> float:
    tokens = _tokenise(reply)
    if not tokens:
        return 0.0
    self_hits = sum(1 for token in tokens if token in SELF_PRONOUNS)
    other_hits = sum(1 for token in tokens if token in SECOND_PRONOUNS)
    collective_hits = sum(1 for token in tokens if token in COLLECTIVE_PRONOUNS)
    sensation_hits = len(SELF_SENSATION_PATTERN.findall(reply or ""))
    introspective_hits = _introspective_hits(reply)
    focus_self = self_hits + (0.9 * sensation_hits)
    if introspective_hits:
        focus_self += 0.35 * introspective_hits
    lead_token = tokens[0]
    if lead_token in SELF_PRONOUNS:
        focus_self += 0.4
    elif lead_token in SECOND_PRONOUNS or lead_token in COLLECTIVE_PRONOUNS:
        other_hits += 1
    repeated_self = sum(1 for token in tokens[:4] if token in SELF_PRONOUNS)
    if repeated_self >= 3:
        focus_self += 0.2
    focus_other = (other_hits * 1.25) + (collective_hits * 0.9) + 0.8
    total = focus_self + focus_other
    if total <= 0.0:
        return 0.0
    score = focus_self / total
    return max(0.0, min(score, 1.0))


def score_response(
    user_message: str,
    ai_reply: str,
    *,
    tracker: ReinforcementTracker | None = None,
) -> Dict[str, float]:
    """Score a response using heuristic reinforcement metrics."""
    tracker = tracker or ReinforcementTracker()
    user_tokens = _tokenise(user_message)
    reply_tokens = _tokenise(ai_reply)
    user_sentiment = _sentiment_score(user_message)
    reply_sentiment = _sentiment_score(ai_reply)
    valence_delta = reply_sentiment - user_sentiment
    affect_intimacy = _affect_intimacy_score(user_message, ai_reply)
    affect_tension = _affect_tension_score(user_message, ai_reply)
    tension_penalty = 0.45 * max(affect_tension, 0.0)
    play_bonus = 0.3 * max(-affect_tension, 0.0)
    affect_valence = max(-1.0, min(reply_sentiment - tension_penalty + play_bonus, 1.0))

    length_ratio = _length_ratio(user_message, ai_reply)
    length_score = max(0.0, min(length_ratio, 2.0))

    engagement_score = _entropy(reply_tokens)

    outward_streak = _outward_focus_streak(user_tokens, reply_tokens, tracker)
    self_focus = _self_preoccupation(ai_reply)
    authenticity = _authenticity_score(ai_reply, outward_streak, self_focus)
    assistant_drift = _assistant_drift(ai_reply)

    return {
        "valence_delta": round(valence_delta, 4),
        "length_score": round(length_score, 4),
        "engagement_score": round(engagement_score, 4),
        "authenticity_score": round(authenticity, 4),
        "assistant_drift": round(assistant_drift, 4),
        "self_preoccupation": round(self_focus, 4),
        "affect_valence": round(affect_valence, 4),
        "affect_intimacy": round(affect_intimacy, 4),
        "affect_tension": round(affect_tension, 4),
        "outward_streak_score": round(outward_streak, 4),
    }


__all__ = ["score_response", "reset_outward_streak", "ReinforcementTracker"]




