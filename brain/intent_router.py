"""Lightweight intent routing using a handcrafted linear classifier."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
INTENTS = ("emotional", "analytical", "narrative", "reflective")
FEATURE_LABELS = {
    "__question__": "question punctuation",
    "__short__": "short utterance",
    "__long__": "extended turn",
}

# Hand-tuned weights trained offline on project-specific transcripts.
INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "emotional": {
        "__bias__": -0.6,
        "feel": 1.6,
        "feeling": 1.3,
        "feelings": 1.3,
        "overwhelmed": 1.8,
        "sad": 1.5,
        "happy": 1.2,
        "angry": 1.2,
        "anxious": 1.6,
        "anxiety": 1.4,
        "cope": 1.5,
        "support": 0.9,
        "comfort": 1.0,
        "lonely": 1.6,
        "upset": 1.3,
        "calm": 0.8,
        "encourage": 0.9,
    },
    "analytical": {
        "__bias__": -0.4,
        "__question__": 1.0,
        "__short__": -0.2,
        "why": 1.6,
        "how": 1.5,
        "what": 0.9,
        "explain": 1.7,
        "analysis": 1.6,
        "calculate": 1.5,
        "compare": 1.3,
        "evaluate": 1.3,
        "steps": 1.2,
        "process": 1.1,
        "plan": 0.9,
        "metric": 1.0,
        "metrics": 1.0,
        "strategy": 1.0,
        "debug": 1.2,
    },
    "narrative": {
        "__bias__": -1.2,
        "story": 1.8,
        "stories": 1.6,
        "tell": 1.2,
        "imagine": 1.6,
        "scenario": 1.4,
        "narrative": 1.5,
        "character": 1.3,
        "plot": 1.4,
        "describe": 1.1,
        "scene": 1.1,
        "journey": 1.2,
        "adventure": 1.2,
        "setting": 1.0,
    },
    "reflective": {
        "__bias__": -0.9,
        "__short__": 0.8,
        "__long__": 0.2,
        "reflect": 1.7,
        "reflection": 1.6,
        "lesson": 1.4,
        "lessons": 1.3,
        "learned": 1.4,
        "insight": 1.3,
        "insights": 1.2,
        "thoughts": 1.2,
        "introspect": 1.6,
        "review": 1.0,
        "consider": 0.9,
        "perspective": 0.9,
    },
}


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    rationale: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _collect_features(text: str, weight: float) -> dict[str, float]:
    tokens = _tokenize(text)
    if not tokens and not text:
        return {}
    features: dict[str, float] = {}
    for token in tokens:
        features[token] = min(features.get(token, 0.0) + weight, 3.0)
    if "?" in text:
        features["__question__"] = min(features.get("__question__", 0.0) + weight, 3.0)
    word_count = len(tokens)
    if word_count <= 4:
        features["__short__"] = min(features.get("__short__", 0.0) + weight, 3.0)
    if word_count >= 12:
        features["__long__"] = min(features.get("__long__", 0.0) + weight, 3.0)
    return features


def _merge_features(*feature_sets: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for feature_map in feature_sets:
        for token, value in feature_map.items():
            merged[token] = min(merged.get(token, 0.0) + value, 3.0)
    return merged


def _describe_feature(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def predict_intent(user_message: str, *, context_summary: str | None = None) -> IntentPrediction:
    """Predict conversational intent using a lightweight linear classifier."""
    primary_features = _collect_features(user_message, 1.0)
    context_features = _collect_features(context_summary or "", 0.5) if context_summary else {}
    features = _merge_features(primary_features, context_features)
    if not features:
        return IntentPrediction("analytical", 0.1, "no signal")

    scores: dict[str, float] = {}
    evidences: dict[str, list[tuple[float, str]]] = {}
    for intent in INTENTS:
        weights = INTENT_WEIGHTS[intent]
        score = weights.get("__bias__", 0.0)
        contributions: list[tuple[float, str]] = []
        for token, magnitude in features.items():
            weight = weights.get(token)
            if weight:
                score += weight * min(magnitude, 1.0)
                contributions.append((weight, _describe_feature(token)))
        scores[intent] = score
        contributions.sort(reverse=True)
        evidences[intent] = contributions[:3]

    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]
    exp_scores = {intent: math.exp(score - max_score) for intent, score in scores.items()}
    total = sum(exp_scores.values())
    probabilities = {intent: exp / total for intent, exp in exp_scores.items()} if total else {intent: 0.0 for intent in INTENTS}
    confidence = probabilities.get(best_intent, 0.0)
    rationale_tokens = [label for _, label in evidences.get(best_intent, []) if label]
    rationale = f"signals: {', '.join(rationale_tokens)}" if rationale_tokens else "bias preference"
    return IntentPrediction(best_intent, confidence, rationale)


__all__ = ["IntentPrediction", "predict_intent"]
