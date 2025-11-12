from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

HORMONE_NAMES: Sequence[str] = ["dopamine", "serotonin", "cortisol", "oxytocin", "noradrenaline"]
REINFORCEMENT_KEYS: Sequence[str] = [
    "valence_delta",
    "length_score",
    "engagement_score",
    "authenticity_score",
    "assistant_drift",
    "self_preoccupation",
    "affect_valence",
    "affect_intimacy",
    "affect_tension",
]


def load_turns(log_path: Path) -> List[dict]:
    if not log_path.exists():
        raise FileNotFoundError(f"Endocrine log file not found: {log_path}")
    turns: List[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            pre = payload.get("pre", {})
            post = payload.get("post", {})
            if not pre or not post:
                continue
            pre_h = pre.get("hormones") or {}
            post_h = post.get("hormones") or {}
            if not all(name in pre_h and name in post_h for name in HORMONE_NAMES):
                continue
            reinforcement = payload.get("reinforcement") or {}
            turns.append(
                {
                    "pre": pre_h,
                    "post": post_h,
                    "reinforcement": reinforcement,
                    "intent": payload.get("intent", "unknown"),
                    "length_label": payload.get("length_label", ""),
                    "profile": payload.get("profile", ""),
                }
            )
    return turns


def build_feature_map(turns: Iterable[dict]) -> Tuple[List[str], Dict[str, int]]:
    intents = sorted({turn["intent"] for turn in turns if turn.get("intent")})
    length_labels = sorted({turn["length_label"] for turn in turns if turn.get("length_label")})
    profiles = sorted({turn["profile"] for turn in turns if turn.get("profile")})

    feature_names: List[str] = ["bias"]
    feature_names += [f"h_pre_{name}" for name in HORMONE_NAMES]
    feature_names += [f"reinforcement_{key}" for key in REINFORCEMENT_KEYS]
    feature_names += [f"intent::{intent}" for intent in intents]
    feature_names += [f"length::{label}" for label in length_labels]
    feature_names += [f"profile::{profile}" for profile in profiles]

    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    return feature_names, feature_index


def encode_features(
    turn: dict,
    feature_index: Dict[str, int],
    *,
    hormone_scale: float,
) -> np.ndarray:
    x = np.zeros(len(feature_index), dtype=np.float64)
    x[feature_index["bias"]] = 1.0
    pre_h = turn["pre"]
    for name in HORMONE_NAMES:
        idx = feature_index.get(f"h_pre_{name}")
        if idx is not None:
            x[idx] = float(pre_h.get(name, 0.0)) / hormone_scale
    reinforcement = turn.get("reinforcement") or {}
    for key in REINFORCEMENT_KEYS:
        idx = feature_index.get(f"reinforcement_{key}")
        if idx is not None:
            x[idx] = float(reinforcement.get(key, 0.0))
    intent = turn.get("intent")
    if intent:
        idx = feature_index.get(f"intent::{intent}")
        if idx is not None:
            x[idx] = 1.0
    length_label = turn.get("length_label")
    if length_label:
        idx = feature_index.get(f"length::{length_label}")
        if idx is not None:
            x[idx] = 1.0
    profile = turn.get("profile")
    if profile:
        idx = feature_index.get(f"profile::{profile}")
        if idx is not None:
            x[idx] = 1.0
    return x


def build_dataset(turns: List[dict], feature_index: Dict[str, int], hormone_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for turn in turns:
        pre = turn["pre"]
        post = turn["post"]
        pre_vector = encode_features(turn, feature_index, hormone_scale=hormone_scale)
        delta = np.array([float(post[name]) - float(pre[name]) for name in HORMONE_NAMES], dtype=np.float64)
        xs.append(pre_vector)
        ys.append(delta)
    if not xs:
        raise ValueError("No usable samples found in endocrine log.")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def train_model(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    validation_ratio: float,
    rng: random.Random,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    sample_count = xs.shape[0]
    indices = list(range(sample_count))
    rng.shuffle(indices)
    split = max(1, int(sample_count * (1.0 - validation_ratio)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < sample_count else []

    x_train = xs[train_idx]
    y_train = ys[train_idx]
    if len(val_idx) > 0:
        x_val = xs[val_idx]
        y_val = ys[val_idx]
    else:
        x_val = None
        y_val = None

    weights, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)

    metrics: Dict[str, Dict[str, float]] = {"train": {}, "validation": {}}
    train_pred = x_train @ weights
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2, axis=0))
    train_mae = np.mean(np.abs(train_pred - y_train), axis=0)
    metrics["train"]["rmse"] = train_rmse.tolist()
    metrics["train"]["mae"] = train_mae.tolist()

    if x_val is not None and y_val is not None and len(val_idx) > 0:
        val_pred = x_val @ weights
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2, axis=0))
        val_mae = np.mean(np.abs(val_pred - y_val), axis=0)
        metrics["validation"]["rmse"] = val_rmse.tolist()
        metrics["validation"]["mae"] = val_mae.tolist()
        metrics["validation"]["samples"] = len(val_idx)
    else:
        metrics["validation"]["rmse"] = []
        metrics["validation"]["mae"] = []
        metrics["validation"]["samples"] = 0

    metrics["train"]["samples"] = len(train_idx)
    return weights, metrics


def serialize_model(
    feature_names: Sequence[str],
    weights: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    *,
    hormone_scale: float,
) -> dict:
    return {
        "feature_names": list(feature_names),
        "hormone_names": list(HORMONE_NAMES),
        "weights": {
            hormone: weights[:, idx].tolist() for idx, hormone in enumerate(HORMONE_NAMES)
        },
        "normalization": {
            "hormone_scale": hormone_scale,
        },
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hormone dynamics model from endocrine logs.")
    parser.add_argument("--log-file", default="logs/endocrine_turns.jsonl", help="Path to endocrine log file.")
    parser.add_argument("--output", default="config/hormone_model.json", help="Destination for the trained model.")
    parser.add_argument("--hormone-scale", type=float, default=100.0, help="Normalization factor for hormone values.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Fraction of samples reserved for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument("--dry-run", action="store_true", help="Run training without writing the output model.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    turns = load_turns(log_path)
    if len(turns) < 10:
        raise SystemExit(f"Not enough samples to train (found {len(turns)}, need at least 10).")

    feature_names, feature_index = build_feature_map(turns)
    xs, ys = build_dataset(turns, feature_index, hormone_scale=args.hormone_scale)

    rng = random.Random(args.seed)
    weights, metrics = train_model(xs, ys, validation_ratio=args.validation_ratio, rng=rng)

    model = serialize_model(feature_names, weights, metrics, hormone_scale=args.hormone_scale)
    output_path = Path(args.output)
    if args.dry_run:
        print(f"[dry-run] Skipping write to {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(f"Saved hormone model to {output_path}")
    print("Training samples:", metrics["train"]["samples"])
    if metrics["validation"]["samples"]:
        print("Validation samples:", metrics["validation"]["samples"])
    print("Train RMSE:", ", ".join(f"{name}={val:.3f}" for name, val in zip(HORMONE_NAMES, metrics["train"]["rmse"])))
    if metrics["validation"]["rmse"]:
        print(
            "Validation RMSE:",
            ", ".join(f"{name}={val:.3f}" for name, val in zip(HORMONE_NAMES, metrics["validation"]["rmse"])),
        )


if __name__ == "__main__":
    main()
