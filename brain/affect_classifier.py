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
# Optional DirectML acceleration (AMD/Intel GPUs on Windows)
try:
    import torch_directml  # type: ignore[unused-ignore]
except Exception:  # pragma: no cover - optional dependency
    torch_directml = None
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
    safety: float | None = None
    arousal: float | None = None
    approach_avoid: float | None = None
    inhibition_social: float | None = None
    inhibition_vulnerability: float | None = None
    inhibition_self_restraint: float | None = None
    expectedness: str | None = None
    momentum_delta: str | None = None
    intent: tuple[str, ...] | None = None
    affection_subtype: str | None = None
    rpe: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "valence": round(self.valence, 4),
            "intimacy": round(self.intimacy, 4),
            "tension": round(self.tension, 4),
            "confidence": round(self.confidence, 4),
            "tags": list(self.tags),
        }
        optional_fields = {
            "safety": self.safety,
            "arousal": self.arousal,
            "approach_avoid": self.approach_avoid,
            "inhibition_social": self.inhibition_social,
            "inhibition_vulnerability": self.inhibition_vulnerability,
            "inhibition_self_restraint": self.inhibition_self_restraint,
            "expectedness": self.expectedness,
            "momentum_delta": self.momentum_delta,
            "intent": list(self.intent) if self.intent else None,
            "affection_subtype": self.affection_subtype,
            "rpe": self.rpe,
            "rationale": self.rationale,
        }
        for key, value in optional_fields.items():
            if value is not None and value != []:
                payload[key] = value
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
        metadata = {"source": "lora", "raw_completion": None}
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
        if start == -1:
            return None
        if end == -1 or end <= start:
            # attempt to salvage by appending a brace
            snippet = completion[start:] + "}"
            reasoning_tail = ""
        else:
            snippet = completion[start : end + 1]
            reasoning_tail = completion[end + 1 :].strip() if end + 1 < len(completion) else ""
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            pass
        else:
            try:
                intim = data.get("intimacy", data.get("intimity", 0.0))
                tags = data.get("tags") if isinstance(data.get("tags"), list) else []
                if len(tags) > 5:
                    tags = tags[:5]
                return {
                    "valence": float(data.get("valence", 0.0)),
                    "intimacy": float(intim),
                    "tension": float(data.get("tension", 0.0)),
                    "confidence": float(data.get("confidence", 0.85)),
                    "tags": tags,
                    "reasoning": None,
                    "rationale": None,
                }
            except (TypeError, ValueError):
                return None
        # regex fallback for incomplete JSON
        import re
        def _grab(key, default=0.0):
            m = re.search(rf'"{key}"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+)', completion)
            return float(m.group(1)) if m else default
        val = _grab("valence", 0.0)
        intim = _grab("intimacy", _grab("intimity", 0.0))
        tens = _grab("tension", 0.0)
        conf = _grab("confidence", 0.0)
        return {
            "valence": val,
            "intimacy": intim,
            "tension": tens,
            "confidence": conf,
            "tags": [],
            "reasoning": None,
            "rationale": None,
        }


class TorchHeadAffectClassifier(AffectClassifier):
    """Classifier that runs the trained regression heads directly (base + LoRA + head.pt)."""

    def __init__(self, config: Mapping[str, Any]):
        fallback_config = config.get("fallback_rules") or {}
        super().__init__(fallback_config)
        if not torch or not AutoTokenizer or not AutoModelForCausalLM or not PeftModel:
            raise RuntimeError("Transformers/PEFT are required for the torch-head affect classifier.")
        try:
            from fine_tune.train_affect_lora import (
                MultiHeadAffect,
                INTENTS,
                EXPECTEDNESS,
                MOMENTUM,
                AFFECTION_SUB,
            )
        except Exception as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(f"Unable to import training heads: {exc}") from exc

        self._base_model_path = config.get("base_model_path")
        self._adapter_path = config.get("adapter_path")
        self._head_path = config.get("head_path")
        if not self._base_model_path or not self._adapter_path or not self._head_path:
            raise ValueError("torch_head config must include base_model_path, adapter_path, head_path.")

        self._device_pref = (config.get("device") or "auto").lower()
        self._model_confidence = float(config.get("model_confidence", 0.9))
        self._max_length = int(config.get("max_length", 256))
        self._intents = INTENTS
        self._expectedness = EXPECTEDNESS
        self._momentum = MOMENTUM
        self._aff_sub = AFFECTION_SUB

        self._tokenizer = None
        self._model: Any = None
        self._device: torch.device | Any = None
        self._load_error: Exception | None = None

    def _select_device(self) -> torch.device | Any:
        if self._device_pref == "dml" or (self._device_pref == "auto" and torch_directml):
            try:
                return torch_directml.device()  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._device_pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self._device_pref == "cpu":
            return torch.device("cpu")
        # auto fallback
        return torch.device("cpu")

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_error:
            raise self._load_error
        device = self._select_device()
        # Use fp16 weights by default (saves memory/bandwidth); DirectML will keep them on the GPU.
        dtype = torch.float16
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._base_model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                self._base_model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            base = PeftModel.from_pretrained(base, self._adapter_path)
            # training class
            from fine_tune.train_affect_lora import MultiHeadAffect

            model = MultiHeadAffect(base, base.config.hidden_size)
            state = torch.load(self._head_path, map_location="cpu", weights_only=False)
            head_state = state.get("head") if isinstance(state, dict) else state
            model.load_state_dict(head_state, strict=False)

            model.to(device, dtype=dtype)
            model.eval()
        except Exception as exc:
            self._load_error = exc
            raise

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def classify(self, text: str) -> AffectClassification:
        if not text.strip():
            return super().classify(text)
        try:
            self._ensure_loaded()
        except Exception:
            return super().classify(text)

        t0 = time.time()
        toks = self._tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        toks = {k: v.to(self._device) for k, v in toks.items()}
        with torch.no_grad():
            out = self._model(**toks)
        latency = (time.time() - t0) * 1000.0

        axes = out["axes"][0].detach().cpu().tolist()
        val, intim, tens = axes
        misc = out["misc"][0].detach().cpu().tolist()
        inh_self = float(out["inh_self"][0].detach().cpu().item())

        expected_idx = int(out["expectedness"][0].argmax().item())
        momentum_idx = int(out["momentum"][0].argmax().item())
        aff_idx = int(out["aff_sub"][0].argmax().item())

        intent_logits = out["intent"][0].detach()
        intent_probs = torch.sigmoid(intent_logits).cpu().tolist()
        intents = tuple(lbl for lbl, p in zip(self._intents, intent_probs) if p >= 0.35)

        def clamp(x, lo, hi):
            return max(lo, min(hi, float(x)))

        payload = {
            "valence": clamp(val, -1.0, 1.0),
            "intimacy": clamp(intim, 0.0, 1.0),
            "tension": clamp(tens, -1.0, 1.0),
            "confidence": self._model_confidence,
            "tags": _derive_tags(val, intim, tens, text.lower()),
            "safety": clamp(misc[3], -1.0, 1.0),
            "arousal": clamp(misc[2], -1.0, 1.0),
            "approach_avoid": clamp(misc[4], -1.0, 1.0),
            "inhibition_social": clamp(misc[6], 0.0, 1.0),
            "inhibition_vulnerability": clamp(misc[7], 0.0, 1.0),
            "inhibition_self_restraint": clamp(inh_self, 0.0, 1.0),
            "expectedness": self._expectedness[expected_idx],
            "momentum_delta": self._momentum[momentum_idx],
            "affection_subtype": self._aff_sub[aff_idx],
            "rpe": clamp(misc[5], -1.0, 1.0),
            "intent": intents,
        }
        metadata = {
            "source": "torch_head",
            "latency_ms": round(latency, 1),
            "device": str(self._device),
        }
        return AffectClassification(
            valence=payload["valence"],
            intimacy=payload["intimacy"],
            tension=payload["tension"],
            confidence=payload["confidence"],
            tags=tuple(payload["tags"]),
            safety=payload["safety"],
            arousal=payload["arousal"],
            approach_avoid=payload["approach_avoid"],
            inhibition_social=payload["inhibition_social"],
            inhibition_vulnerability=payload["inhibition_vulnerability"],
            inhibition_self_restraint=payload["inhibition_self_restraint"],
            expectedness=payload["expectedness"],
            momentum_delta=payload["momentum_delta"],
            intent=payload["intent"],
            affection_subtype=payload["affection_subtype"],
            rpe=payload["rpe"],
            metadata=metadata,
        )


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
        self._soft_timeout = float(config.get("soft_timeout", min(self._timeout, 1.0)))
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
            raw_debug = None
            if os.environ.get("AFFECT_DEBUG_RAW"):
                raw_debug = completion[:240]
            metadata = {
                "source": "llama_cpp",
                "raw_completion": raw_debug,
                "latency_ms": round(latency * 1000, 1),
                "reasoning": parsed.get("reasoning"),
                "rationale": parsed.get("rationale"),
            }
            return AffectClassification(
                valence=parsed.get("valence", 0.0),
                intimacy=parsed.get("intimacy", 0.0),
                tension=parsed.get("tension", 0.0),
                confidence=parsed.get("confidence", self._model_confidence),
                tags=tuple(parsed.get("tags") or ()),
                safety=parsed.get("safety"),
                arousal=parsed.get("arousal"),
                approach_avoid=parsed.get("approach_avoid"),
                inhibition_social=parsed.get("inhibition_social"),
                inhibition_vulnerability=parsed.get("inhibition_vulnerability"),
                inhibition_self_restraint=parsed.get("inhibition_self_restraint"),
                expectedness=parsed.get("expectedness"),
                momentum_delta=parsed.get("momentum_delta"),
                intent=tuple(parsed.get("intent") or ()),
                affection_subtype=parsed.get("affection_subtype"),
                rpe=parsed.get("rpe"),
                rationale=parsed.get("rationale"),
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

        # Fast probe; give a bit more than half a second to avoid flapping on busy boots.
        probe_timeout = min(self._timeout, 1.2)
        try:
            response = self._client.get(models_url, timeout=probe_timeout)
            if response.status_code == 200:
                return
        except Exception:
            pass

        raise TimeoutError(f"affect llama.cpp sidecar not reachable at {models_url}")

    def _query_model(self, text: str) -> str:
        assert self._client is not None
        prompt = (
            "You are an affect scorer. Respond with exactly one JSON object using these keys: "
            "valence (-1..1), intimacy (0..1), tension (-1..1), safety (-1..1), arousal (-1..1), "
            "approach_avoid (-1..1), inhibition_social (0..1), inhibition_vulnerability (0..1), "
            "inhibition_self_restraint (0..1), expectedness (expected|mild_surprise|strong_surprise), "
            "momentum_delta (with_trend|soft_turn|hard_turn), intent (array: reassure|comfort|flirt_playful|dominate|apologize|boundary|manipulate|deflect|vent|inform|seek_support), "
            "affection_subtype (warm|forced|defensive|sudden|needy|playful|manipulative|overwhelmed|intimate|confused|none), "
            "rpe (-1..1), confidence (0..1), tags (array; use []). "
            "Keep rationale optional and under 30 words if included. Do not default everything to zero unless the input is truly neutral/empty. "
            "Even brief or ambiguous inputs should receive small non-zero scores that reflect tone (e.g., +/-0.05..0.15) and lower confidence when uncertain. "
            "Example:\n"
            "Input: I feel calm and close to you.\n"
            "JSON: {\"valence\":0.4,\"intimacy\":0.55,\"tension\":0.05,\"safety\":0.4,\"arousal\":0.1,\"approach_avoid\":0.35,\"inhibition_social\":0.1,\"inhibition_vulnerability\":0.15,\"inhibition_self_restraint\":0.1,\"expectedness\":\"expected\",\"momentum_delta\":\"with_trend\",\"intent\":[\"reassure\"],\"affection_subtype\":\"warm\",\"rpe\":0.2,\"confidence\":0.7,\"tags\":[]}\n"
            "Now score this input without any extra text.\n"
            "Input: "
            + text.strip()
            + "\nJSON:"
        )
        payload = {
            "model": self._alias,
            "prompt": prompt,
            "temperature": self._temperature,
            "max_tokens": max(self._max_new_tokens, 32),
            "stop": ["\nInput:", "Input:", "\n\n"],
        }
        resp = self._client.post(
            f"http://{self._host}:{self._port}/v1/completions",
            json=payload,
            timeout=max(self._soft_timeout, self._timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from affect llama server.")
        return (choices[0].get("text") or "").strip()

    @staticmethod
    def _parse_completion(completion: str) -> dict[str, Any] | None:
        if not completion:
            return None
        start = completion.find("{")
        if start == -1:
            return None
        end = completion.rfind("}")
        snippet = completion[start : end + 1] if end != -1 else completion[start:] + "}"
        if snippet.count("{") > snippet.count("}"):
            snippet = snippet + "}"
        data = None
        try:
            data = json.loads(snippet)
        except Exception:
            data = None
        # regex fallback
        if data is None:
            import re

            def grab(key, default=0.0):
                m = re.search(rf'"{key}"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+)', completion)
                return float(m.group(1)) if m else default
            val = grab("valence", 0.0)
            intim = grab("intimacy", grab("intimity", 0.0))
            tens = grab("tension", 0.0)
            conf = grab("confidence", 0.65)
            return {
                "valence": val,
                "intimacy": intim,
                "tension": tens,
                "confidence": conf,
                "tags": [],
                "safety": grab("safety", None),
                "arousal": grab("arousal", None),
                "approach_avoid": grab("approach_avoid", None),
                "inhibition_social": grab("inhibition_social", None),
                "inhibition_vulnerability": grab("inhibition_vulnerability", None),
                "inhibition_self_restraint": grab("inhibition_self_restraint", None),
                "rpe": grab("rpe", None),
                "reasoning": None,
                "rationale": None,
            }
        try:
            intim = data.get("intimacy", data.get("intimity", 0.0))
            tags_raw = data.get("tags") if isinstance(data.get("tags"), list) else []
            tags = [str(tag)[:48] for tag in tags_raw if isinstance(tag, (str, int, float))]
            if len(tags) > 5:
                tags = tags[:5]
            intents_raw = data.get("intent") if isinstance(data.get("intent"), list) else []
            intent_allowed = {
                "reassure",
                "comfort",
                "flirt_playful",
                "dominate",
                "apologize",
                "boundary",
                "manipulate",
                "deflect",
                "vent",
                "inform",
                "seek_support",
            }
            intents = tuple(
                lbl for lbl in (str(x).strip() for x in intents_raw) if lbl in intent_allowed
            )[:5]
            expectedness_allowed = {"expected", "mild_surprise", "strong_surprise"}
            momentum_allowed = {"with_trend", "soft_turn", "hard_turn"}
            affection_allowed = {
                "warm",
                "forced",
                "defensive",
                "sudden",
                "needy",
                "playful",
                "manipulative",
                "overwhelmed",
                "intimate",
                "confused",
                "none",
            }
            return {
                "valence": float(data.get("valence", 0.0)),
                "intimacy": float(intim),
                "tension": float(data.get("tension", 0.0)),
                "confidence": float(data.get("confidence", 0.85)),
                "tags": tags,
                "safety": _maybe_float(data.get("safety")),
                "arousal": _maybe_float(data.get("arousal")),
                "approach_avoid": _maybe_float(data.get("approach_avoid")),
                "inhibition_social": _maybe_float(data.get("inhibition_social")),
                "inhibition_vulnerability": _maybe_float(data.get("inhibition_vulnerability")),
                "inhibition_self_restraint": _maybe_float(data.get("inhibition_self_restraint")),
                "expectedness": data.get("expectedness") if data.get("expectedness") in expectedness_allowed else None,
                "momentum_delta": data.get("momentum_delta") if data.get("momentum_delta") in momentum_allowed else None,
                "intent": intents,
                "affection_subtype": data.get("affection_subtype") if data.get("affection_subtype") in affection_allowed else None,
                "rpe": _maybe_float(data.get("rpe")),
                "rationale": _trim_text(data.get("rationale"), 160),
            }
        except Exception:
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
    if cls_type == "torch_head":
        try:
            return TorchHeadAffectClassifier(config)
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


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _trim_text(value: Any, max_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value[:max_chars]


def _is_empty_payload(payload: Mapping[str, Any] | None) -> bool:
    """Detect degenerate model outputs so we can fall back to heuristics."""
    if not payload:
        return True
    numeric_keys = [
        "valence",
        "intimacy",
        "tension",
        "confidence",
        "safety",
        "arousal",
        "approach_avoid",
        "rpe",
    ]
    if any(abs(float(payload.get(k, 0.0) or 0.0)) > 0.005 for k in numeric_keys):
        return False
    if payload.get("tags"):
        return False
    if payload.get("intent"):
        return False
    return True


__all__ = [
    "AffectClassifier",
    "AffectClassification",
    "LoraAffectClassifier",
    "TorchHeadAffectClassifier",
    "LlamaCppAffectClassifier",
    "load_affect_classifier",
]
