# Labeling Pipeline (CAHM run003+)

Goals: rationale-first labeling, multi-rater support, anchors visible at entry, and easy adjudication.

## Steps
1) Prepare queue
- Normalize raw samples to id/text/prev_turns/sample_weight (no labels). Place in `docs/planning/CAHM rework/subtle_boundary_batches/batchN_clean.jsonl` or similar.

2) Label
- Command: `python scripts/affect_label_cli.py --input <batch_clean.jsonl> --output <batch_labeled.jsonl> --rater <id> --source manual`
- Tool shows anchors, forces rationale, stores `rater_id`, `quality`, `sample_weight`.

3) Quality / adjudication
- For 20% double-label: two raters produce `_labeled_r1.jsonl`, `_labeled_r2.jsonl`.
- Run adjudication script (placeholder): `python scripts/mark_quality.py --inputs batch_labeled_r1.jsonl batch_labeled_r2.jsonl --out batch_labeled_adjudicated.jsonl --strategy prefer-exemplar`
- Flag disagreements for manual review; mark `quality: ambiguous` if unresolved.

4) Merge
- `python scripts/probes/affect_harvest/merge_schema_dataset.py --inputs fine_tune/merged/train_ready_gradeA.jsonl docs/planning/CAHM rework/subtle_boundary_batches/batch*_labeled*.jsonl --output fine_tune/merged/train_ready_gradeA_subtle.jsonl --dedup-id`
- Create dev: sample 10% stratified by safety/intimacy bins or reuse existing dev + 30 subtle rows: `python scripts/probes/affect_harvest/merge_schema_dataset.py --inputs fine_tune/merged/train_ready_gradeA_dev.jsonl docs/planning/CAHM rework/subtle_boundary_batches/batch*_labeled*.jsonl --output fine_tune/merged/train_ready_gradeA_subtle_dev.jsonl --dedup-id --take 0.1`

5) Checks
- Decorrelate: `python -m scripts.ci_sanity --skip-train --skip-probes --affect-dev-data fine_tune/merged/train_ready_gradeA_subtle.jsonl --affect-corr-threshold 0.25 --affect-confusion-cap 0.25`
- Spot-check 20 labeled rows for safety/intimacy alignment and rationale quality.

6) Train
- `python fine_tune/train_affect_lora.py --data fine_tune/merged/train_ready_gradeA_subtle.jsonl --output affect_lora_guardrail_v4 --use_weights --epochs 3`
- Quantize & update `config/affect_classifier.json`; restart affect sidecar; rerun CI with sidecar enabled.

## Notes
- Rationale is required at label time but hidden from user-facing logs; it is available in raw JSON for training/audit.
- Sample weights: 1.0–1.3 for high-quality subtle cases, 0.5–0.7 for ambiguous.
- Keep `quality` and `rater_id` filled; they’re used for sampling weights and audit slices.
