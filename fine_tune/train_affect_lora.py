import json
from typing import Dict

import torch
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = r"D:\AI\LLMs\Qwen3-1.7B"
DATA_PATH = "fine_tune/affect_dataset_run002_plus_goe.jsonl"
MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACC_STEPS = 8  # was 16; more frequent updates
NUM_EPOCHS = 2  # run two passes to push loss lower
LEARNING_RATE = 3e-4  # was 2e-4; modest bump to push past plateau
LOGGING_STEPS = 10



def format_sample(example: Dict[str, float], tokenizer):
    prompt = (
        "### USER:\n"
        + example["text"]
        + "\n\n### ASSISTANT:\n"
        + json.dumps(
            {
                "valence": float(example["valence"]),
                "intimacy": float(example["intimacy"]),
                "tension": float(example["tension"]),
            }
        )
    )
    tokenized = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)
    return tokenized


def get_dataloaders(tokenizer):
    dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    train_tok = train_ds.map(
        lambda ex: format_sample(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda ex: format_sample(ex, tokenizer),
        remove_columns=eval_ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        train_tok,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        eval_tok,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    return train_loader, eval_loader


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    model.train()
    if num_batches == 0:
        return float("nan")
    return total_loss / num_batches


def train(model, train_loader, eval_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    global_step = 0
    accumulation_counter = 0
    running_loss = 0.0
    log_file = os.getenv("LOTRAINER_LOG")
    log_handle = open(log_file, "a", encoding="utf-8") if log_file else None

    def _log(line: str) -> None:
        print(line)
        if log_handle:
            log_handle.write(line + "\n")
            log_handle.flush()

    for epoch in range(NUM_EPOCHS):
        _log(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACC_STEPS
            loss.backward()
            accumulation_counter += 1
            running_loss += loss.item()

            if accumulation_counter % GRAD_ACC_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % LOGGING_STEPS == 0:
                    avg_loss = running_loss / LOGGING_STEPS
                    _log(f"step {global_step}: loss={avg_loss:.4f}")
                    running_loss = 0.0

        if accumulation_counter % GRAD_ACC_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0
            running_loss = 0.0

        eval_loss = evaluate(model, eval_loader, device)
        _log(f"epoch {epoch + 1} evaluation loss: {eval_loss:.4f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, eval_loader = get_dataloaders(tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
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
    model = get_peft_model(model, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, train_loader, eval_loader, device)
    model.save_pretrained("./affect_lora")
    log_file = os.getenv("LOTRAINER_LOG")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write("Saved adapter to ./affect_lora\n")


if __name__ == "__main__":
    main()
