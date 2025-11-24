"""
Multitask affect training with LoRA on Qwen3 (or compatible causal LM).
- Predicts continuous axes (valence, intimacy, tension), arousal/safety/approach, inhibition triplet, sincerity/playfulness, rpe.
- Predicts categorical fields (expectedness, momentum_delta, affection_subtype) and multi-label intents.
- Uses a pooled last-token representation with small heads; combines losses plus axis decorrelation penalty.

CLI:
  python fine_tune/train_affect_lora.py --data fine_tune/harvest/run003_external_rescored.weighted.jsonl --output affect_lora_run003

Environment:
  LOTRAINER_LOG (optional) to append training logs.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL = os.getenv("AFFECT_LORA_BASE", r"D:/AI/LLMs/Qweb3-1.7B for training")
DEFAULT_DATA = r"fine_tune/merged/train_ready_guardrail_v6_plus_subtle.cleaned.jsonl"
INTENTS = [
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
]
EXPECTEDNESS = ["expected", "mild_surprise", "strong_surprise"]
MOMENTUM = ["with_trend", "soft_turn", "hard_turn"]
AFFECTION_SUB = [
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
]


@dataclass
class Sample:
    text: str
    prev_turns: List[str]
    labels: Dict
    weight: float


class AffectDataset(Dataset):
    def __init__(self, path: Path):
        self.samples: List[Sample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            self.samples.append(
                Sample(
                    text=ex["text"],
                    prev_turns=ex.get("prev_turns") or [],
                    labels=ex,
                    weight=float(ex.get("sample_weight", 1.0)),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: Sequence[Sample], tokenizer, device, mask_prob: float):
    texts = []
    for s in batch:
        ctx = ""
        if s.prev_turns:
            for turn in s.prev_turns[-2:]:
                ctx += f"{turn}\n"
        ctx += s.text
        texts.append(ctx)
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}

    def m(label):
        return torch.tensor(label, dtype=torch.float32, device=device)

    labels = [s.labels for s in batch]
    out = {
        "valence": m([l["valence"] for l in labels]),
        "intimacy": m([l["intimacy"] for l in labels]),
        "tension": m([l["tension"] for l in labels]),
        "sincerity": m([l.get("sincerity", 0.8) for l in labels]),
        "playfulness": m([l.get("playfulness", 0.2) for l in labels]),
        "arousal": m([l.get("arousal", 0.0) for l in labels]),
        "safety": m([l.get("safety", 0.0) for l in labels]),
        "approach": m([l.get("approach_avoid", 0.0) for l in labels]),
        "rpe": m([l.get("rpe", 0.0) for l in labels]),
        "inh_social": m([l.get("inhibition", {}).get("social", 0.3) for l in labels]),
        "inh_vuln": m([l.get("inhibition", {}).get("vulnerability", 0.5) for l in labels]),
        "inh_self": m([l.get("inhibition", {}).get("self_restraint", 0.4) for l in labels]),
        "expectedness": torch.tensor(
            [EXPECTEDNESS.index(l.get("expectedness", "expected")) for l in labels],
            dtype=torch.long,
            device=device,
        ),
        "momentum": torch.tensor(
            [MOMENTUM.index(l.get("momentum_delta", "with_trend")) for l in labels],
            dtype=torch.long,
            device=device,
        ),
        "aff_sub": torch.tensor(
            [AFFECTION_SUB.index(l.get("affection_subtype", "none")) for l in labels],
            dtype=torch.long,
            device=device,
        ),
        "intent": torch.tensor(
            [
                [1.0 if i in l.get("intent", []) else 0.0 for i in INTENTS]
                for l in labels
            ],
            dtype=torch.float32,
            device=device,
        ),
        "weights": m([s.weight for s in batch]),
    }

    # mask some fields (masked-field supervision) to fight coupling
    mask = torch.rand(len(batch), device=device)
    for key in ("valence", "intimacy", "tension"):
        drop = mask < mask_prob
        out[key] = out[key].masked_fill(drop, float("nan"))

    return toks, out


class MultiHeadAffect(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int):
        super().__init__()
        self.base = base_model
        self.reg_axes = nn.Linear(hidden_size, 3)  # valence, intimacy, tension
        self.reg_misc = nn.Linear(hidden_size, 8)  # sincerity, playfulness, arousal, safety, approach, rpe, inh_social, inh_vuln
        self.reg_inh_self = nn.Linear(hidden_size, 1)
        self.expectedness = nn.Linear(hidden_size, len(EXPECTEDNESS))
        self.momentum = nn.Linear(hidden_size, len(MOMENTUM))
        self.aff_sub = nn.Linear(hidden_size, len(AFFECTION_SUB))
        self.intent = nn.Linear(hidden_size, len(INTENTS))

    def forward(self, input_ids, attention_mask):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1][:, -1, :]  # last token rep
        return {
            "axes": self.reg_axes(hidden),
            "misc": self.reg_misc(hidden),
            "inh_self": self.reg_inh_self(hidden),
            "expectedness": self.expectedness(hidden),
            "momentum": self.momentum(hidden),
            "aff_sub": self.aff_sub(hidden),
            "intent": self.intent(hidden),
        }


def decorrelation_penalty(pred_axes):
    # pred_axes: (B,3)
    # need sensible batch size and variance; otherwise corrcoef returns NaN.
    if pred_axes.size(0) < 4:
        return torch.tensor(0.0, device=pred_axes.device)
    c = torch.corrcoef(pred_axes.T)
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    # zero diag
    off_diag = c - torch.diag(torch.diag(c))
    return (off_diag ** 2).mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    snapshot_epochs = set()
    if args.save_epochs:
        snapshot_epochs = {
            int(e.strip())
            for e in args.save_epochs.split(",")
            if e.strip().isdigit()
        }
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}. Set --model or AFFECT_LORA_BASE to a local model directory.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = AffectDataset(Path(args.data))
    val_split = max(1, int(args.val_split * len(dataset)))
    split_gen = torch.Generator().manual_seed(args.seed)
    train_set, eval_set = torch.utils.data.random_split(
        dataset, [len(dataset) - val_split, val_split], generator=split_gen
    )

    def _sampler(ds):
        if args.use_weights:
            weights = torch.tensor([s.weight for s in ds], dtype=torch.double)
            return WeightedRandomSampler(
                weights,
                num_samples=len(ds),
                replacement=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
        return RandomSampler(ds, generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=_sampler(train_set),
        collate_fn=lambda b: collate_fn(b, tokenizer, device, args.mask_prob),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        sampler=RandomSampler(eval_set, generator=torch.Generator().manual_seed(args.seed)),
        collate_fn=lambda b: collate_fn(b, tokenizer, device, 0.0),
    )

    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base = get_peft_model(base, lora_config)
    hidden_size = base.config.hidden_size
    model = MultiHeadAffect(base, hidden_size).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    log = open(os.getenv("LOTRAINER_LOG"), "a", encoding="utf-8") if os.getenv("LOTRAINER_LOG") else None

    def _log(msg):
        print(msg)
        if log:
            log.write(msg + "\n")
            log.flush()

    def _save_model(out_path: Path):
        out_path.mkdir(parents=True, exist_ok=True)
        model.base.save_pretrained(out_path / "adapter")
        torch.save({"head": model.state_dict(), "intents": INTENTS}, out_path / "head.pt")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for toks, lbl in train_loader:
            pred = model(**toks)
            losses = []
            # regression axes with mask (nan = skip)
            for i, key in enumerate(["valence", "intimacy", "tension"]):
                target = lbl[key]
                mask = ~torch.isnan(target)
                if mask.any():
                    losses.append(mse(pred["axes"][mask, i], target[mask]))
            # misc regression
            misc_targets = torch.stack(
                [
                    lbl["sincerity"],
                    lbl["playfulness"],
                    lbl["arousal"],
                    lbl["safety"],
                    lbl["approach"],
                    lbl["rpe"],
                    lbl["inh_social"],
                    lbl["inh_vuln"],
                ],
                dim=1,
            )
            losses.append(mse(pred["misc"], misc_targets))
            losses.append(mse(pred["inh_self"].squeeze(-1), lbl["inh_self"]))
            # categorical
            losses.append(ce(pred["expectedness"], lbl["expectedness"]))
            losses.append(ce(pred["momentum"], lbl["momentum"]))
            losses.append(ce(pred["aff_sub"], lbl["aff_sub"]))
            # intents
            losses.append(bce(pred["intent"], lbl["intent"]))
            # decorrelation on predicted axes
            losses.append(args.decorr_lambda * decorrelation_penalty(pred["axes"]))

            loss = sum(losses) / len(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (global_step + 1) % args.accum_steps == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1

            if global_step % args.log_steps == 0:
                _log(f"step {global_step} loss {loss.item():.4f}")

        # eval
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for toks, lbl in eval_loader:
                pred = model(**toks)
                l = mse(pred["axes"], torch.stack([lbl["valence"], lbl["intimacy"], lbl["tension"]], dim=1))
                eval_losses.append(l.item())
        eval_mse = sum(eval_losses) / len(eval_losses)
        epoch_idx = epoch + 1
        _log(f"epoch {epoch_idx}/{args.epochs} eval mse (axes): {eval_mse:.4f}")
        if epoch_idx in snapshot_epochs:
            snap_dir = Path(f"{args.output}_epoch{epoch_idx}")
            _save_model(snap_dir)
            _log(f"saved snapshot to {snap_dir}")

    out_dir = Path(args.output)
    _save_model(out_dir)
    _log(f"saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", default="affect_lora_guardrail_v3")
    # tuned for ~3.7k rows; effective batch 32 on 16GB cards
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--mask_prob", type=float, default=0.15, help="probability to drop axis labels (masked-field supervision)")
    p.add_argument("--decor_lambda", dest="decorr_lambda", type=float, default=0.1)
    p.add_argument("--use_weights", action="store_true", help="use sample_weight if present")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="force device selection")
    p.add_argument("--save_epochs", type=str, default="", help="comma-separated epoch numbers to snapshot adapter/head")
    p.add_argument("--seed", type=int, default=42, help="seed for dataset split, samplers, and masking")
    p.add_argument("--val_split", type=float, default=0.1, help="fraction of data for validation (0<val_split<1)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
