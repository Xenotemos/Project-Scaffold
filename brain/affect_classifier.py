"""Lightweight affect classifier for user prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import asyncio
import json
import os
import re
import time
from typing import Any, Mapping, Sequence

# Optional heavy imports for model-backed classifier
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    PeftModel = None
try:
    import httpx  # type: ignore[unused-ignore]
except Exception:  # pragma: no cover - optional dependency
    httpx = None
from brain.affect_sidecar import AffectSidecarManager


POSITIVE_WORDS = {
    "appreciate",
    "care",
    "cheerful",
    "connected",
    "gentle",
    "glad",
    "grateful",
    "happy",
    "hope",
    "hopeful",
    "joy",
    "joyful",
    "kind",
    "light",
    "love",
    "loved",
    "lovely",
    "peace",
    "peaceful",
    "playful",
    "proud",
    "relaxed",
    "relief",
    "safe",
    "secure",
    "support",
    "supportive",
    "tender",
    "warm",
}

NEGATIVE_WORDS = {
    "ache",
    "aching",
    "alone",
    "angry",
    "anxious",
    "ashamed",
    "confused",
    "depressed",
    "drained",
    "fear",
    "frightened",
    "frustrated",
    "hurt",
    "lonely",
    "numb",
    "panic",
    "sad",
    "scared",
    "shaky",
    "stressed",
    "tense",
    "tired",
    "upset",
    "worried",
}

AFFECTION_WORDS = {
    "affection",
    "affectionate",
    "love",
    "loved",
    "loving",
    "luv",
    "babe",
    "baby",
    "beloved",
    "caring",
    "cherish",
    "darling",
    "dear",
    "dearie",
    "dearest",
    "honey",
    "sweetie",
    "sweetness",
    "hug",
    "kiss",
    "lover",
    "mwah",
    "sweet",
    "sweetheart",
    "tenderness",
}

DIMINUTIVE_WORDS = {
    "little",
    "tiny",
    "small",
    "softie",
    "kitten",
    "bud",
}

TENSION_WORDS = {
    "ache",
    "aching",
    "clench",
    "clenched",
    "clenching",
    "rigid",
    "strain",
    "strained",
    "stress",
    "stressed",
    "tight",
    "tighten",
    "tightness",
    "tense",
    "tension",
    "wound",
}

EMOJI_WARM = {"❤️", "💕", "💖", "🥰", "😘", "😍"}
EMOJI_STRESS = {"😰", "😟", "😢", "😭", "😔", "😩", "😖", "😤"}

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class AffectClassification:
    """Container for a single user-turn affect classification."""

    valence: float
    intimacy: float
    tension: float
    confidence: float
    tags: tuple[str, ...]
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "valence": round(self.valence, 4),
            "intimacy": round(self.intimacy, 4),
            "tension": round(self.tension, 4),
            "confidence": round(self.confidence, 4),
            "tags": list(self.tags),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class AffectClassifier:
    """Rule-based affect classifier that can be configured via JSON."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._valence_cap = float(config.get("valence_cap", 4.0))
        self._intimacy_cap = float(config.get("intimacy_cap", 4.0))
        self._tension_cap = float(config.get("tension_cap", 4.0))
        self._emoji_weight = float(config.get("emoji_weight", 1.5))
        self._punctuation_bonus = float(config.get("punctuation_bonus", 0.2))
        self._confidence_scale = float(config.get("confidence_scale", 3.0))

    def classify(self, text: str) -> AffectClassification:
        stripped = (text or "").strip()
        tokens = _tokenise(stripped)
        lowered = stripped.lower()
        positive_hits = _count_members(tokens, POSITIVE_WORDS)
        negative_hits = _count_members(tokens, NEGATIVE_WORDS)
        affection_hits = _count_members(tokens, AFFECTION_WORDS)
        diminutive_hits = _count_members(tokens, DIMINUTIVE_WORDS)
        tension_hits = _count_members(tokens, TENSION_WORDS)

        emoji_warm = sum(1 for ch in text if ch in EMOJI_WARM)
        emoji_stress = sum(1 for ch in text if ch in EMOJI_STRESS)
        affection_hits += int(self._emoji_weight * emoji_warm)
        tension_hits += int(self._emoji_weight * emoji_stress)

        exclamation_bonus = self._punctuation_bonus if "!" in text else 0.0

        short_boost = 0.0
        if len(tokens) <= 2 and affection_hits:
            short_boost += 0.45 * affection_hits
        if len(tokens) <= 2 and tension_hits:
            short_boost -= 0.35 * tension_hits
        valence_score = _score_directional(
            positive_hits,
            negative_hits,
            self._valence_cap,
        )
        valence_score = max(-1.0, min(1.0, valence_score + short_boost))
        intimacy_score = min(
            1.0,
            (affection_hits + 0.5 * diminutive_hits) / self._intimacy_cap,
        )
        if affection_hits and len(tokens) <= 3:
            intimacy_score = min(1.0, intimacy_score + 0.18 * affection_hits)
        tension_score = min(
            1.0,
            (tension_hits + max(0.0, -exclamation_bonus)) / self._tension_cap,
        )

        if len(tokens) <= 2 and any(ch in stripped for ch in "?!") and valence_score > 0:
            valence_score = min(1.0, valence_score + 0.15)
        if not tokens and stripped:
            valence_score = 0.0
            intimacy_score = 0.0
            tension_score = 0.0

        signal_strength = (
            abs(valence_score)
            + intimacy_score
            + tension_score
            + (positive_hits + negative_hits + affection_hits + tension_hits + len(tokens) * 0.2) * 0.05
        )
        confidence = max(
            0.0,
            min(1.0, signal_strength / max(1.0, self._confidence_scale)),
        )
        if affection_hits:
            confidence = max(confidence, min(1.0, 0.2 + 0.12 * affection_hits))
        tags = _derive_tags(valence_score, intimacy_score, tension_score, lowered)

        metadata = {
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "affection_hits": affection_hits,
            "tension_hits": tension_hits,
        }
        return AffectClassification(
            valence=valence_score,
            intimacy=intimacy_score,
            tension=tension_score,
            confidence=confidence,
            tags=tuple(tags),
            metadata=metadata,
        )


class LoraAffectClassifier(AffectClassifier):
    """Classifier that queries a LoRA-tuned causal LM for affect scores."""

    def __init__(self, config: Mapping[str, Any]):
        fallback_config = config.get("fallback_rules") or {}
        super().__init__(fallback_config)
        if not torch or not AutoTokenizer or not AutoModelForCausalLM or not PeftModel:
            raise RuntimeError("Transformers/PEFT are required for the LoRA affect classifier.")

        self._base_model_path = config.get("base_model_path")
        self._adapter_path = config.get("adapter_path")
        if not self._base_model_path or not self._adapter_path:
            raise ValueError("LoRA config must include 'base_model_path' and 'adapter_path'.")

        self._max_new_tokens = int(config.get("max_new_tokens", 64))
        self._temperature = float(config.get("temperature", 0.1))
        self._top_p = float(config.get("top_p", 0.9))
        self._model_confidence = float(config.get("model_confidence", 0.85))
        self._device_pref = (config.get("device") or "auto").lower()

        self._tokenizer = None
        self._model = None
        self._device: torch.device | None = None
        self._load_error: Exception | None = None

    def classify(self, text: str) -> AffectClassification:
        try:
            self._ensure_loaded()
        except Exception:
            return super().classify(text)

        prompt = self._format_prompt(text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        completion = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = self._parse_completion(completion)
        if not parsed:
            return super().classify(text)
        metadata = {"source": "lora", "raw_completion": completion}
        return AffectClassification(
            valence=parsed.get("valence", 0.0),
            intimacy=parsed.get("intimacy", 0.0),
            tension=parsed.get("tension", 0.0),
            confidence=parsed.get("confidence", self._model_confidence),
            tags=tuple(parsed.get("tags") or ()),
            metadata=metadata,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_error:
            raise self._load_error

        if self._device_pref == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self._device_pref == "cpu":
            device = torch.device("cpu")
        elif self._device_pref == "auto":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device("cpu")
        self._device = device

        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._base_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self._base_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, self._adapter_path)
            model.to(device)
            model.eval()
        except Exception as exc:
            self._load_error = exc
            raise

        self._tokenizer = tokenizer
        self._model = model

    @staticmethod
    def _format_prompt(text: str) -> str:
        clean = text.strip()
        return f"### USER:\n{clean}\n\n### ASSISTANT:\n"

    @staticmethod
    def _parse_completion(completion: str) -> dict[str, Any] | None:
        if not completion:
            return None
        start = completion.find("{")
        end = completion.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = completion[start : end + 1]
        reasoning_tail = completion[end + 1 :].strip() if end + 1 < len(completion) else ""
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return None
        try:
            return {
                "valence": float(data.get("valence", 0.0)),
                "intimacy": float(data.get("intimacy", 0.0)),
                "tension": float(data.get("tension", 0.0)),
                "confidence": float(data.get("confidence", 0.85)),
                "tags": data.get("tags") or [],
                "reasoning": reasoning_tail or data.get("reasoning"),
            }
        except (TypeError, ValueError):
            return None


class LlamaCppAffectClassifier(AffectClassifier):
    """Classifier that calls a dedicated llama.cpp server running the GGUF affect head."""

    def __init__(self, config: Mapping[str, Any]):
        fallback_config = config.get("fallback_rules") or {}
        super().__init__(fallback_config)
        if httpx is None:  # pragma: no cover - runtime guard
            raise RuntimeError("httpx is required for the llama.cpp affect classifier.")

        self._binary = Path(config.get("llama_server_bin") or "")
        self._model_path = Path(config.get("model_path") or "")
        if not self._binary or not self._model_path:
            raise ValueError("llama_cpp config requires 'llama_server_bin' and 'model_path'.")

        self._host = str(config.get("host") or "127.0.0.1")
        self._port = int(config.get("port") or 8082)
        self._alias = str(config.get("alias") or "affect")
        self._extra_args: list[str] = list(config.get("extra_args") or ())
        self._max_new_tokens = int(config.get("max_new_tokens", 64))
        self._temperature = float(config.get("temperature", 0.0))
        self._top_p = float(config.get("top_p", 0.9))
        self._model_confidence = float(config.get("model_confidence", 0.85))
        self._timeout = float(config.get("timeout", 20.0))
        self._readiness_timeout = float(config.get("readiness_timeout", 30.0))
        self._start_server = bool(config.get("start_server", True))

        self._client: httpx.Client | None = None
        self._last_error: Exception | None = None
        self._sidecar = AffectSidecarManager(config)

    def classify(self, text: str) -> AffectClassification:
        try:
            if not text.strip():
                return super().classify(text)
            start = time.time()
            self._ensure_ready()
            completion = self._query_model(text)
            parsed = self._parse_completion(completion)
            latency = time.time() - start
            if not parsed:
                return super().classify(text)
            metadata = {
                "source": "llama_cpp",
                "raw_completion": completion,
                "latency_ms": round(latency * 1000, 1),
                "reasoning": parsed.get("reasoning"),
            }
            return AffectClassification(
                valence=parsed.get("valence", 0.0),
                intimacy=parsed.get("intimacy", 0.0),
                tension=parsed.get("tension", 0.0),
                confidence=parsed.get("confidence", self._model_confidence),
                tags=tuple(parsed.get("tags") or ()),
                metadata=metadata,
            )
        except Exception:
            return super().classify(text)

    def _ensure_ready(self) -> None:
        if self._last_error:
            raise self._last_error
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        models_url = f"http://{self._host}:{self._port}/v1/models"

        # Fast, low-timeout probe; sidecar should already be started at app startup.
        probe_timeout = min(self._timeout, 0.6)
        try:
            response = self._client.get(models_url, timeout=probe_timeout)
            if response.status_code == 200:
                return
        except Exception:
            pass

        raise TimeoutError(f"affect llama.cpp sidecar not reachable at {models_url}")

    def _query_model(self, text: str) -> str:
        assert self._client is not None
        system_prompt = (
            "You are an affect scorer. Return ONLY a JSON object with keys "
            'valence (float -1..1), intimacy (float 0..1), tension (float 0..1), '
            'confidence (float 0..1), and tags (array of strings). No prose.'
        )
        payload = {
            "model": self._alias,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text.strip()},
            ],
            "stream": False,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": max(16, self._max_new_tokens),
        }
        response = self._client.post(
            f"http://{self._host}:{self._port}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from affect llama server.")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        return content if content else reasoning

    @staticmethod
    def _parse_completion(completion: str) -> dict[str, Any] | None:
        if not completion:
            return None
        start = completion.find("{")
        end = completion.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = completion[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return None
        try:
            return {
                "valence": float(data.get("valence", 0.0)),
                "intimacy": float(data.get("intimacy", 0.0)),
                "tension": float(data.get("tension", 0.0)),
                "confidence": float(data.get("confidence", 0.85)),
                "tags": data.get("tags") or [],
            }
        except (TypeError, ValueError):
            return None


def load_affect_classifier(config_path: str | Path | None = None) -> AffectClassifier:
    """Load an AffectClassifier from JSON config or fall back to defaults."""
    if not config_path:
        return AffectClassifier()
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return AffectClassifier()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AffectClassifier()

    cls_type = (config.get("type") or "heuristic").lower()
    if cls_type == "lora":
        try:
            return LoraAffectClassifier(config)
        except Exception:
            return AffectClassifier(config.get("fallback_rules") or {})
    if cls_type in {"llama_cpp", "gguf"}:
        try:
            return LlamaCppAffectClassifier(config)
        except Exception:
            return AffectClassifier(config.get("fallback_rules") or {})
    return AffectClassifier(config)


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _count_members(tokens: Sequence[str], vocab: set[str]) -> int:
    return sum(1 for token in tokens if token in vocab)


def _score_directional(
    positive_hits: int,
    negative_hits: int,
    cap: float,
) -> float:
    numerator = positive_hits - negative_hits
    denominator = max(cap, 1.0)
    return max(-1.0, min(1.0, numerator / denominator))


def _derive_tags(
    valence: float,
    intimacy: float,
    tension: float,
    lowered_text: str,
) -> list[str]:
    tags: list[str] = []
    if valence >= 0.35:
        tags.append("positive")
    elif valence <= -0.35:
        tags.append("negative")
    if intimacy >= 0.2:
        tags.append("affectionate")
    if tension >= 0.2:
        tags.append("tense")
    if "?" in lowered_text:
        tags.append("curious")
    return tags


__all__ = [
    "AffectClassifier",
    "AffectClassification",
    "LoraAffectClassifier",
    "LlamaCppAffectClassifier",
    "load_affect_classifier",
]
