"""Offline staging probe for mistral-inference with a minimal prompt."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from mistral_inference.generate import generate, generate_mamba
from mistral_inference.main import get_model_cls, load_tokenizer
from mistral_inference.mamba import Mamba
from mistral_inference.transformer import Transformer
from mistral_inference.xformers_compat import XFORMERS_AVAILABLE


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_generation(
    model: Transformer | Mamba,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    eos_id: int | None,
) -> Tuple[list[int], float]:
    """Generate tokens and measure elapsed wall time."""
    generate_fn = generate if isinstance(model, Transformer) else generate_mamba

    _sync_device(model.device)
    start = time.perf_counter()
    if generate_fn is generate:
        generated_tokens, _ = generate_fn(
            [prompt_tokens],
            model,
            [[]],
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=eos_id,
        )
    else:
        generated_tokens, _ = generate_fn(
            [prompt_tokens],
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=eos_id,
        )
    _sync_device(model.device)
    elapsed = time.perf_counter() - start
    return (generated_tokens[0] if generated_tokens else []), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage mistral-inference locally with a minimal prompt.")
    parser.add_argument("--config", default="config/mistral_inference_stage.json", help="Path to staging config JSON.")
    parser.add_argument("--model-path", help="Override model path.")
    parser.add_argument("--prompt", help="Override prompt text.")
    parser.add_argument("--max-tokens", type=int, help="Override max new tokens.")
    parser.add_argument("--temperature", type=float, help="Override temperature.")
    parser.add_argument("--warmup-runs", type=int, help="Override number of warmup runs.")
    args = parser.parse_args()

    config = _load_config(Path(args.config))

    model_path = Path(args.model_path or config["model_path"]).expanduser()
    prompt_text = args.prompt or config.get("prompt", "")
    max_tokens = args.max_tokens or int(config.get("max_tokens", 128))
    temperature = args.temperature or float(config.get("temperature", 0.6))
    warmup_runs = args.warmup_runs or int(config.get("warmup_runs", 1))

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if model_path.is_file():
        raise ValueError(
            f"Expected a model directory with params.json/tokenizer files, received file path: {model_path}"
        )

    tokenizer_wrapper = load_tokenizer(model_path)
    tokenizer = tokenizer_wrapper.instruct_tokenizer.tokenizer
    prompt_tokens = tokenizer.encode(prompt_text, bos=True, eos=False)

    model_cls = get_model_cls(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = model_cls.from_folder(model_path, max_batch_size=1, num_pipeline_ranks=1, device=device, dtype=dtype)

    eos_id = tokenizer.eos_id if hasattr(tokenizer, "eos_id") else None

    if not XFORMERS_AVAILABLE:
        print("[warn] xformers not available; falling back to PyTorch attention kernels.", file=sys.stderr, flush=True)

    for _ in range(max(warmup_runs, 0)):
        _run_generation(model, prompt_tokens, max_tokens=8, temperature=0.0, eos_id=eos_id)

    generated, elapsed = _run_generation(model, prompt_tokens, max_tokens, temperature, eos_id)
    total_new_tokens = len(generated)
    tok_per_sec = (total_new_tokens / elapsed) if elapsed > 0 and total_new_tokens else 0.0

    result = {
        "device": str(model.device),
        "dtype": str(model.dtype),
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": total_new_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tok_per_sec,
        "output_text": tokenizer.decode(generated),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
