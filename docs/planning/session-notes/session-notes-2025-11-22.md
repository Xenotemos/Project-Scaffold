# Session Notes — 2025-11-22

## What we did
- Added three new labeled bridge chunks (GPT/Gemini synthetic slices) plus a 346-row safe/positive pull; cleaned mojibake, fixed empty/nested text, and normalized schema/weights.
- Built combined labeled pools:
  - `synthetic_bridge_all_labeled.jsonl` (1,297 rows after dedup of parts 1–3).
  - Safe-positive pull labeled: 432 rows.
  - Legacy labeled chunks 1–11 and synthetic GPT set retained.
- Tried multiple merges with/without backfill; backfill continues to raise val–int correlations:
  - Labeled-only (v8 earlier) gave val–int ≈0.16 but only ~1.2k rows.
  - Full labeled pool (3,814 uniques) still shows val–int ≈0.51 even with modest intimacy/safety caps and ambiguous down-weighting.
  - Backfill at 0.01–0.02 weight and trims (1k rows dropped) didn’t move val–int; backfill clearly dominates correlation when included.
- Current artifacts:
  - Labeled-only “grade A” split (no backfill): `fine_tune/merged/train_ready_gradeA.jsonl` / `..._dev.jsonl` (3,432 / 382). Weighted corr: val–int ~0.54 after legacy chunks included.
  - Full labeled snapshot: `docs/planning/CAHM rework/labeled/all_labeled_gradeA.jsonl` (3,814 uniques).
  - Backfill-trimmed files: `affect_run003_schema_backfill_trim2000.jsonl`, `..._trim3000.jsonl` (dropped 1k lowest-weight rows).

## Observations
- Legacy labeled chunks (all_labeled_merged_v6) carry baked-in valence–intimacy coupling; adding bridge/safe-positive helps but doesn’t overcome this unless legacy rows are heavily down-weighted or intimacy-capped harder.
- Safety distribution still negative-skewed (median around -0.13 to -0.2) when legacy rows dominate; bridge and positive-pull data are safety-neutral/positive.
- Backfill, even at very low weight, reintroduces higher val–int; for decorrelation we should train labeled-only or drastically shrink/clip backfill weights.

## Options / Next actions
1) **Decorrelate without losing data**
   - Aggressive intimacy cap on positive lines when safety <0.2 (e.g., cap at 0.2) and down-weight ambiguous legacy rows (e.g., ×0.2) before final merge.
   - Keep backfill out for the main run; optionally do a light second-stage fine-tune with backfill at 0.005 and cap backfill weights at 1–2.
2) **Add counter-weight data**
   - Label the remaining safe/positive pulls if any; generate another ~1k targeted safe/positive/low-intimacy + neutral, and add inhibition-contrast pairs to pull val–int down.
3) **Training choice**
   - If we need low correlation now, train on labeled-only (bridge + legacy but with stronger caps/down-weights) and skip backfill.
   - If we need maximum diversity, accept higher val–int or keep backfill for a brief second-stage.

## File map (latest)
- Labeled pools: `docs/planning/CAHM rework/labeled/all_labeled_gradeA.jsonl`, `synthetic_bridge_all_labeled.jsonl`, `cahm_candidates_positivepull_labeled.jsonl`.
- Train-ready (no backfill): `fine_tune/merged/train_ready_gradeA.jsonl`, `train_ready_gradeA_dev.jsonl`.
- Backfill trimmed: `fine_tune/harvest/affect_run003_schema_backfill_trim2000.jsonl`, `..._trim3000.jsonl`.
- High-correlation merge with backfill (reference): `affect_run003_schema_train_v9/v10/v11.jsonl` in `fine_tune/merged/`.

## Decisions needed
- Choose strategy: labeled-only with stronger decorrelation vs. minimal backfill.
- Approve more aggressive intimacy/safety normalization + ambiguous down-weighting on legacy rows.
- Whether to generate/label another ~1k counter-weight samples to balance val–int and safety.

