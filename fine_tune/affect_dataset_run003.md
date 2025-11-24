
# Affect Dataset Run003 Plan

## Current Baseline (run002 + GoEmotions)
- Samples: 5,817
- Mean valence: **+0.142** (skewed positive)
- Mean intimacy: **+0.472** (most turns read as close)
- Mean tension: **+0.448**
- Only 0 examples with intimacy ≤ –0.1 (no “distant” coverage)
- Positive valence (≥0.2): 2,852 rows
- Negative valence (≤ –0.2): 1,987 rows
- High intimacy (≥0.3): 3,924 rows
- High tension (≥0.4): 2,627 rows
- Top tags still warmth-centric: affection/care dominate

## Target Archetypes & Counts
| Archetype | Description | Target New Samples |
|-----------|-------------|--------------------|
| Hostile Intimacy | Angry/jealous lovers, negative valence but high intimacy & tension | 600 |
| Aggressive Distance | Low intimacy, high tension, sharp/hostile language | 600 |
| Neutral-but-close | Flat valence (~0), moderate/high intimacy, low tension | 500 |
| Flat/Numb | Low valence & low tension (“burned out”, “numb”) | 400 |
| Fearful/Urgent | High tension regardless of valence (panic, fear) | 400 |
| Playful Neutral | Light teasing, moderate intimacy, near-zero valence | 300 |
| Safety Edge Cases | Content referencing boundaries, discomfort | 200 |
Total new labels: ~3,000 (brings dataset to ~8.8k rows)

## Labeling Guidelines
- Assign valence/intimacy/tension independently (use –1.0 to +1.0). Avoid mirroring old heuristics.
- Capture confidence 0–1 from rater.
- Tag archetype (`hostile_intimacy`, `flat_numb`, etc.) plus existing taxonomy (`tension`, `affection`, etc.).
- Include provenance: `source` (`manual_chat`, `goemotions`, `synthetic_probe`), `session_id` or `file`.
- Optional: store raw conversation snippet (up to last 2 turns) under `context` for future multi-turn prompts.

## Harvest Pipeline
1. **Manual transcripts:** run a script that scans `docs/logs/webui/interactions.log` + `logs/affect_head_raw.jsonl` to list candidates by tag (e.g., tension). Export to `fine_tune/harvest/candidates_manual.jsonl`.
2. **GoEmotions filters:** pull anger, annoyance, disgust, cautious, apprehension categories (balanced). Map to archetypes with heuristics (e.g., anger+love ⇒ hostile intimacy).
3. **Synthetic probes:** write short scripts in `scripts/probes/scenarios/affect_archetypes/` that produce targeted dialogues for missing cells (flat/numb, fearful). Capture the user utterances and label manually.
4. **Label tool:** use lightweight CLI (TBD) or a spreadsheet + `scripts/probes/export_affect_dataset.py` to convert to JSONL with the schema:
```json
{
  "text": "...",
  "valence": 0.4,
  "intimacy": -0.2,
  "tension": 0.7,
  "confidence": 0.6,
  "tags": ["hostile_intimacy", "argument"],
  "source": "manual_chat",
  "context": ["prev turn", "current turn"]
}
```

## Integration Steps
1. Save raw new labels under `fine_tune/harvest/run003/`.
2. Concatenate `run002_plus_goe` + new labels → `affect_dataset_run003.jsonl`.
3. Shuffle & split (train/dev/test) with per-archetype stratification.
4. Re-run `train_affect_lora.py` with the new dataset, logging per-axis MAE.
5. Convert to GGUF + rebuild CAHM sidecar.

## Open Tasks
- [ ] Build candidate extraction script for manual chats.
- [ ] Define GoEmotions→archetype mapping file.
- [ ] Draft 3–4 synthetic scenarios per missing archetype.
- [ ] Create simple labeling helper (CLI or Google Sheet schema).
- [ ] Schedule labeling sprint to hit target counts.
