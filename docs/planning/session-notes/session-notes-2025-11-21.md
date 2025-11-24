# Session Notes - 2025-11-21

## Summary
- Gathered and cleaned high-quality labels:
  - 202 core GPT-labeled rows (cahm_label_gpt_2) plus three extra strata chunks (playful/hostile/boundary/neutral-close/flat) labeled and fixed.
  - Combined deduped labeled pool: 548 rows (`docs/planning/CAHM rework/label_combined_marked.jsonl`), quality-weighted (exemplar/clean) and sanity-checked.
- Backfill remains 9k heuristic rows (`affect_run003_schema_backfill.jsonl`); merged train/dev currently 8,757 / 973 rows after including labeled pool.
- Current correlations (merged): val–int ~0.58 (still high), val–ten ~-0.20; labeled pool alone has lower coupling (val–int ~0.24).
- Trainer upgraded (multi-head + decorrelation) ready to run once dataset balance is fixed; training not started.

## Pending / Next Actions
1) Expand labeled pool by ~1,000 additional rows (targeted strata) to strengthen coverage.
2) Downsample/dim the heuristic backfill by ~2,000 rows (or equivalently reduce their weights) to let high-quality labels steer the model.
3) Apply light up-weight to labeled rows (e.g., exemplar×8, clean×5) and modest down-weight to remaining backfill.
4) Re-merge train/dev, validate correlations (goal |val–int| ≲ 0.3), then run `train_affect_lora.py` with the local model path.

## Files touched / artifacts
- Labeled chunks (fixed): `docs/planning/CAHM rework/labeled chunks/chunk1_cahm_labeled_gpt_fixed.jsonl`, `chunk2_cahm_labeled_gpt_fixed.jsonl`, `chunk3_cahm_labeled_gpt_fixed.jsonl`
- Core GPT set: `docs/planning/CAHM rework/cahm_label_gpt_2.jsonl`
- Combined labeled pool (quality-weighted): `docs/planning/CAHM rework/label_combined_marked.jsonl`
- Current train/dev (not yet decorrelated): `fine_tune/affect_run003_schema_train.jsonl`, `fine_tune/affect_run003_schema_dev.jsonl`

## Keep in mind
- Backfill dominance is keeping val–int coupling high; need either more labeled rows or stronger weighting/downsampling before training.
- Safety/intimacy fixes applied to boundary/creep lines in all chunks; dedup applied to chunk3.
- Gold/dev (`affect_gold_labels.jsonl`) remains excluded from train.
