from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

HORMONE_NAMES: Sequence[str] = ["dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"]
TRAIT_FEATURES: Sequence[str] = ["steadiness", "curiosity", "warmth", "tension"]
OUTPUT_NAMES: Sequence[str] = [
    "temperature_delta",
    "top_p_delta",
    "frequency_penalty_delta",
    "presence_penalty_delta",
    "max_tokens_delta",
    "self_bias_scale",
]
SELF_OBSERVATION_WORDS: Dict[str, float] = {
    "notice": 0.34,
    "feel": 0.3,
    "tension": 0.26,
    "breath": 0.24,
    "pulse": 0.22,
    "tight": 0.2,
    "ache": 0.18,
}
CONTROLLER_HORMONE_SCALE = 45.0


def _normalize_hormone(value: float, baseline: float = 50.0) -> float:
    return max(-1.5, min(1.5, (value - baseline) / CONTROLLER_HORMONE_SCALE))


def _ensure_bias(features: Dict[str, float]) -> None:
    features.setdefault("bias", 1.0)


def load_turns(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    turns: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            sampling = payload.get("sampling") or {}
            if not sampling:
                continue
            turns.append(payload)
    return turns


def _derive_features(entry: dict) -> Tuple[Dict[str, float], List[str]]:
    controller_input = entry.get("controller_input") or {}
    feature_map = dict(controller_input.get("features") or {})
    tags = list(controller_input.get("tags") or [])
    if feature_map:
        _ensure_bias(feature_map)
        return feature_map, [str(tag).lower() for tag in tags]

    feature_map = {"bias": 1.0}
    tags = [str(tag).lower() for tag in tags]
    intent = entry.get("intent")
    if intent:
        feature_map[f"intent:{str(intent).lower()}"] = 1.0
    length_label = entry.get("length_label")
    if length_label:
        feature_map[f"length:{str(length_label).lower()}"] = 1.0
    profile = entry.get("profile")
    if profile:
        feature_map[f"profile:{str(profile).lower()}"] = 1.0

    pre = entry.get("pre") or {}
    trait_overview = pre.get("trait_overview") or {}
    for trait in TRAIT_FEATURES:
        feature_map[f"trait:{trait}"] = float(trait_overview.get(trait, 0.0))
    hormones = pre.get("hormones") or {}
    for name in HORMONE_NAMES:
        feature_map[f"hormone:{name}"] = _normalize_hormone(float(hormones.get(name, 50.0)))

    affect_tags = pre.get("affect_tags") or []
    for tag in affect_tags:
        lowered = str(tag).lower()
        if lowered not in tags:
            tags.append(lowered)

    return feature_map, tags


def _derive_targets(entry: dict) -> Dict[str, float]:
    sampling = entry.get("sampling") or {}
    policy_preview = entry.get("policy_preview") or {}
    targets: Dict[str, float] = {}

    def _delta(key: str) -> float:
        final = float(sampling.get(key, 0.0))
        base = float(policy_preview.get(key, final))
        return final - base

    targets["temperature_delta"] = _delta("temperature")
    targets["top_p_delta"] = _delta("top_p")
    targets["frequency_penalty_delta"] = _delta("frequency_penalty")
    targets["presence_penalty_delta"] = _delta("presence_penalty")
    targets["max_tokens_delta"] = float(sampling.get("max_tokens", 0)) - float(policy_preview.get("max_tokens", sampling.get("max_tokens", 0)))

    bias_words = sampling.get("logit_bias_words") or {}
    ratios: List[float] = []
    for word, weight in SELF_OBSERVATION_WORDS.items():
        if abs(weight) < 1e-6:
            continue
        if word in bias_words:
            ratios.append(float(bias_words[word]) / weight)
    targets["self_bias_scale"] = float(sum(ratios) / len(ratios)) if ratios else 0.0
    return targets


def build_feature_space(samples: Sequence[dict]) -> List[str]:
    names: set[str] = {"bias"}
    for sample in samples:
        features = sample["features"]
        tags = sample["tags"]
        names.update(features.keys())
        for tag in tags:
            names.add(f"tag:{tag}")
    return sorted(names)


def assemble_dataset(samples: Sequence[dict], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[List[float]] = []
    y_rows: List[List[float]] = []

    for sample in samples:
        feature_map = dict(sample["features"])
        tags = sample["tags"]
        for tag in tags:
            feature_map.setdefault(f"tag:{tag}", 1.0)
        row = [float(feature_map.get(name, 0.0)) for name in feature_names]
        x_rows.append(row)
        y_rows.append([float(sample["targets"].get(name, 0.0)) for name in OUTPUT_NAMES])

    xs = np.asarray(x_rows, dtype=np.float64)
    ys = np.asarray(y_rows, dtype=np.float64)
    return xs, ys


def train_controller(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Tuple[float, float]]]:
    weights, _, _, _ = np.linalg.lstsq(xs, ys, rcond=None)

    linear_max = np.max(np.abs(xs @ weights)) if xs.size else 0.0
    scale_factor = 1.0
    if linear_max > 2.0:
        scale_factor = 2.0 / linear_max

    weights_input = (scale_factor * weights.T)

    hidden_pre = np.tanh(xs @ (weights * scale_factor))
    raw_outputs = np.tanh(hidden_pre)

    output_scales: Dict[str, float] = {}
    output_bounds: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(OUTPUT_NAMES):
        targets = ys[:, idx]
        raw = raw_outputs[:, idx]
        ratios = [abs(target) / abs(raw_val) for target, raw_val in zip(targets, raw) if abs(raw_val) > 1e-5]
        scale = np.median(ratios) if ratios else 1.0
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        output_scales[name] = float(scale)
        max_target = float(np.max(np.abs(targets))) if targets.size else 0.0
        bound = max_target * 1.2 if max_target > 0 else 0.5
        output_bounds[name] = (-float(bound), float(bound))

    return weights_input, output_scales, output_bounds


def serialize_policy(
    *,
    feature_names: Sequence[str],
    weights_input: np.ndarray,
    output_scales: Dict[str, float],
    output_bounds: Dict[str, Tuple[float, float]],
    state_decay: float,
    recurrent_strength: float,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    hidden_size = len(OUTPUT_NAMES)
    input_flat = weights_input.reshape(hidden_size * len(feature_names)).tolist()
    recurrent = np.eye(hidden_size, dtype=np.float64) * recurrent_strength
    weights_output = np.eye(hidden_size, dtype=np.float64)

    return {
        "version": metadata.get("version", "0.17.4"),
        "metadata": metadata,
        "input_features": list(feature_names),
        "output_names": list(OUTPUT_NAMES),
        "hidden_size": hidden_size,
        "state_decay": state_decay,
        "output_scales": {name: float(scale) for name, scale in output_scales.items()},
        "output_bounds": {name: [float(low), float(high)] for name, (low, high) in output_bounds.items()},
        "weights": {
            "input": input_flat,
            "recurrent": recurrent.reshape(hidden_size * hidden_size).tolist(),
            "hidden_bias": [0.0] * hidden_size,
            "output": weights_output.reshape(hidden_size * hidden_size).tolist(),
            "output_bias": [0.0] * hidden_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controller policy from endocrine logs.")
    parser.add_argument("--log-file", default="logs/endocrine_turns.jsonl", help="Path to the endocrine turn log.")
    parser.add_argument("--output", default="config/controller_policy.json", help="Destination for the trained policy.")
    parser.add_argument("--state-decay", type=float, default=0.6, help="Recurrent state decay factor.")
    parser.add_argument("--recurrent-strength", type=float, default=0.25, help="Diagonal recurrent weight.")
    parser.add_argument("--version", default="0.17.4", help="Version string recorded in metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Run training without writing the output policy.")
    args = parser.parse_args()

    turns = load_turns(Path(args.log_file))
    samples: List[dict] = []
    for turn in turns:
        features, tags = _derive_features(turn)
        targets = _derive_targets(turn)
        samples.append({"features": features, "tags": tags, "targets": targets})

    if len(samples) < 5:
        raise SystemExit(f"Not enough samples to train controller (found {len(samples)}, need at least 5).")

    feature_names = build_feature_space(samples)
    xs, ys = assemble_dataset(samples, feature_names)
    weights_input, output_scales, output_bounds = train_controller(xs, ys)

    metadata = {
        "version": args.version,
        "samples": len(samples),
        "features": len(feature_names),
        "source": Path(args.log_file).as_posix(),
    }
    policy = serialize_policy(
        feature_names=feature_names,
        weights_input=weights_input,
        output_scales=output_scales,
        output_bounds=output_bounds,
        state_decay=max(0.0, min(1.0, args.state_decay)),
        recurrent_strength=float(args.recurrent_strength),
        metadata=metadata,
    )

    output_path = Path(args.output)
    if args.dry_run:
        print(f"[dry-run] Skipping write to {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        print(f"Saved controller policy to {output_path}")
    print(f"Samples used: {len(samples)}")
    print("Output scales:", ", ".join(f"{name}={scale:.3f}" for name, scale in output_scales.items()))


if __name__ == "__main__":
    main()
