# Session Notes - 2025-11-14 Evening

## Completed
- Hardened hormone logic: `_apply_reinforcement_signals` now gates valence/intimacy/tension boosts on actual user affect, gives larger dopamine lifts for affection (especially instruct/base), and explicitly lowers dopamine/oxytocin when tension spikes. Oxytocin/dopamine decay also use two-rate profiles so elevated states linger instead of collapsing immediately (hormones/hormones.py).
- Prevented self-sentiment feedback: `state_engine.register_event` takes `apply_sentiment`; assistant replies, priming hints, and other internal events set it False, so neutral user prompts no longer spike hormones from our own text.
- Added guardrail logit penalties for the leaked safety phrases (“I am not here to run diagnostics…”, “I hear the user as affectionate…”, “Let’s move from this chat log…”). Helper-tone word weights were increased and controller helper penalties adjusted.
- Affect detection upgrade: the rule-based classifier now handles very short inputs (single words, punctuation, emoji) and returns higher-confidence valence/intimacy/tension tags even without the stimulus dropdown. Training scaffolding for a LoRA affect head (Qwen1.5-1.8B Chat) lives in `train_affect_lora.py` + `affect_dataset.jsonl`.
- Diagnostics, probes, and tests: `scripts/diagnostics.py` run is green except known authenticity WARNs; scoped affect-validation probes now show instruct hitting expectations after the retune, base is close (affection + tension now pass after latest tweak). Regression suites (`pytest tests/test_state_engine_affect.py tests/test_reinforcement.py`) pass.

## Things to Keep in Mind
- Qwen LoRA training needs the new deps in `.venv-win` (datasets, torch). Large HF downloads time out easily; keep `temp.md` logs handy for pip install attempts.
- Base profile still has occasional residual guardrail text in manual chats; Voice Guard flags dropped but watch for “Let’s move…” style phrases.
- affects_classifier is still rule-based; the LoRA affect head isn’t trained yet. Once it is, swap `AffectClassifier.classify()` to call the model.

## Next Steps
1. Finish LoRA training: ensure `.venv-win` has `datasets`, `torch`, and rerun `train_affect_lora.py` to train the Qwen affect head. Export adapter + simple inference wrapper.
2. Integrate the trained affect head by replacing the heuristics in `brain/affect_classifier.py` with a call to the LoRA inference function; log outputs to audit accuracy.
3. Re-run full manual chat + affect-validation probes after integration to confirm hormones track user charge without the stimulus dropdown and that base profile hits all expectations.
