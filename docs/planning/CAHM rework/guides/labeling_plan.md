# Labeling Plan (pre-training checklist)

1) Label the queue
- File: `docs/planning/CAHM rework/label_queue.jsonl` (202 rows, deduped from strata: hostile_intimacy, neutral_close, flat_numb, safety_edge, playful_vs_sincere).
- Command: `python scripts/affect_label_cli.py --input "docs/planning/CAHM rework/label_queue.jsonl" --output "docs/planning/CAHM rework/label_queue_labeled.jsonl" --rater <id> --overwrite`
- Goal: mark quality as `clean` or `exemplar`, update all schema fields (expectedness, momentum_delta, intents, inhibition, arousal/safety/approach/rpe, affection_subtype, rationale).

2) Merge labeled data into training pool
- Command after labeling:
```
python scripts/probes/affect_harvest/merge_schema_dataset.py ^
  --inputs fine_tune/harvest/affect_run003_schema_backfill.jsonl "docs/planning/CAHM rework/label_queue_labeled.jsonl" ^
  --exclude_ids "docs/planning/CAHM rework/affect_gold_labels.jsonl" ^
  --train fine_tune/affect_run003_schema_train.jsonl ^
  --dev   fine_tune/affect_run003_schema_dev.jsonl ^
  --dev_ratio 0.1
```
- Re-run validator: `python scripts/affect_dataset_validator.py fine_tune/affect_run003_schema_train.jsonl fine_tune/affect_run003_schema_dev.jsonl --edges=-1.0,-0.7,-0.3,0.3,0.7,1.0 --weight-field=sample_weight`
- Target: drop axis correlations (|val-int|, |val-ten|) toward ≤0.3 after weighting; check charge/momentum coverage.

3) Train after labels merged
- Command (with local model path fixed):
```
python fine_tune/train_affect_lora.py ^
  --model "D:\\AI\\LLMs\\Qwen3-1.7B" ^
  --data fine_tune/affect_run003_schema_train.jsonl ^
  --output affect_lora_run003_schema ^
  --use_weights --epochs 1 --batch_size 2 --log_steps 100
```
- Ensure HF cache resolves the local path (use escaped backslashes or POSIX path).

4) Keep gold/dev separate
- Do not train on `docs/planning/CAHM rework/affect_gold_labels.jsonl`; use it only for eval/regression.
