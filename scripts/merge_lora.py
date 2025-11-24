import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to base HF model directory")
    ap.add_argument("--adapter", required=True, help="Path to PEFT adapter directory")
    ap.add_argument("--head", required=True, help="Path to head.pt with added heads")
    ap.add_argument("--out", required=True, help="Output directory for merged full model")
    args = ap.parse_args()

    # Force CPU merge and disable bitsandbytes to avoid GPU/bnb issues on this host.
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("BNB_CUDA_VERSION", "")

    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        device_map=None,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(
        base,
        args.adapter,
        device_map=None,
        torch_dtype=torch.float32,
        is_trainable=False,
    )
    model = model.merge_and_unload()  # merge LoRA into base weights

    head_path = Path(args.head)
    if head_path.exists():
        state = torch.load(head_path, map_location="cpu")
        head_state = state.get("head") if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(head_state, strict=False)
        print(f"Loaded head state, missing={missing}, unexpected={unexpected}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(out)
    print(f"Merged model saved to {out}")


if __name__ == "__main__":
    main()
