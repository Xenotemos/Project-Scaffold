
# CAHM Reliability & Richer Affect Dataset Plan

## Problems Observed
- Classifier favours warm/affectionate readings because the dataset is dominated by positive/transitional turns.
- Valence, intimacy, and tension scores are tightly coupled (valence↑ almost always predicts intimacy↑) so CAHM rarely emits hostile intimacy or neutral closeness.
- Dataset lacks sustained negative polarity, aggressive, or flat/numb examples, so reinforcement never sees those dynamics.
- Sidecar prompt returns only numeric scores; telemetry never captures reasoning, and CAHM only sees the most recent user turn (no context).
- GGUF Q6_K is responsive on GPU but the prompt is too thin; cutting context length starves the model of the nuance it needs for high-confidence labels.

## Goals
1. Build a balanced affect dataset covering hostile intimacy, neutral-but-close, aggressive/distant, flat/numb, fearful/urgent, and playful neutral turns.
2. Train a new CAHM head that treats valence/intimacy/tension as independent axes; reduce label coupling and add decorrelation terms.
3. Feed richer conversation context (last 1–2 turns) to the sidecar and request short reasoning strings for telemetry.
4. Keep latency <1s by staying on the existing GPU path while restoring enough context length for nuanced reasoning.

## Workstreams

### 1. Dataset Expansion & Relabeling
1. Harvest slices for each missing archetype:
   - Negative valence + high intimacy (jealous, angry, pleading partners)
   - Aggressive / distant (low intimacy, high tension)
   - Neutral but affectionate (close, calm statements without obvious positivity)
   - Flat/numb/under-stimulated (low valence, low tension)
   - Fearful/urgent (high tension regardless of valence)
   - Playful or teasing neutral
2. Target ≥500 labeled examples per archetype; pull from:
   - Manual chat transcripts (select turns where the persona tagged negative or tense)
   - External corpora (GoEmotions “annoyed”, “furious”, “cautious”, etc.)
   - Scripted probes (write short dialogues to seed the extremes)
3. Label valence, intimacy, tension independently for each sample; avoid mirroring the old heuristic patterns.
4. Append the new samples to `fine_tune/affect_dataset_run003.jsonl`, keeping provenance fields so we can slice subsets later.

### 2. Training Pass & Deployment
1. Retrain the LoRA head with the expanded dataset:
   - Separate loss weights per axis and add a decorrelation penalty so valence/intimacy/tension learn orthogonal features.
   - Use existing hyperparameters (LR≈2e-4, grad_acc=8) and log axis-specific validation metrics.
2. Merge the adapter, convert to GGUF (F16 base + Q6_K serving quant), and quantize via llama.cpp tooling.
3. Swap the new GGUF into `config/affect_classifier.json` and restart the sidecar.

### 3. Prompt & Telemetry Upgrades
1. Update the sidecar prompt to include:
   - The last two user turns (or user+assistant for extra context)
   - An explicit request for a `reason` field summarizing how the model interpreted the latest input
2. Capture reasoning text in `brain/affect_classifier` and pipe it through `affect_head_raw.jsonl` / telemetry.
3. Ensure the CAHM console shows both the summary (scores/tags) and the reasoning block so operators can inspect nuance live.

### 4. Validation & Tuning
1. Re-run `python -m scripts.probes.affect_validation --profiles instruct base` and inspect the expectation misses; adjust per-scenario scaling if tension probes still push oxytocin higher than expected.
2. Rerun manual chat QA focusing on hostile/aggressive prompts to confirm CAHM no longer defaults to warm replies and that reasoning text shows the intended nuance.
3. Log before/after histograms of valence/intimacy/tension to verify the axes now span the full range (not just the top-right quadrant).

## Open Questions
- How many manual labels do we need per archetype before diminishing returns?
- Do we need separate calibration curves per profile (instruct vs base) once the dataset broadens?
- Should we add a lightweight confidence-based fallback (e.g., if CAHM confidence <0.25, fall back to heuristics) or raise the minimum confidence thresholds now that variance increases?
