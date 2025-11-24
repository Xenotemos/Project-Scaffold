# Session Notes 2025-11-14-late

## Summary
- `temp.md` captured the affect-head roadmap: dataset requirements (valence/intimacy/tension labels plus categorical tags), model/runtime choices (Qwen3-1.7B-Q8 or Qwen1.5-1.8B-Chat with LoRA/QLoRA adapters), training stack (HF Transformers + PEFT via `train_affect_lora.py`), inference wrapper expectations, integration/fallback strategy for `brain/affect_classifier`, and monitoring/logging for classifier outputs.
- Manual and probe runs logged in `logs/webui/manual_transcript.txt` and `logs/probe_runs/affect_validation.json` show the hormones only react when the UI stimulus toggles, affectionate states decay to neutral within seconds, and helper replies still narrate emotions instead of letting them remain internal.
- Diagnostics still focus on health; the conversation repeatedly asked for a complete, codic-style project sweep that can surface disconnected modules, logic errors, or hormone/affect bugs flagged by the manual/probe observations.

## Keep in mind
- Every user input should reach the classifier so hormones reflect perceived valence/charge; neutral prompts currently raise oxytocin when no stimulus exists, and charged prompts do nothing unless a manual stimulus is set.
- Hormone lifecycles need biologically inspired persistence: some values should stick longer, others should decline slowly, and dopamine/oxytocin transitions must match the input's weight to avoid immoral or unreliable states.
- The entity should stop explaining its feelings; emotional states must stay internal and show up only through subtle cues (a tone shift, a brief admit like "a bit flat") when context calls for it.
- Classifier outputs and hormone traces must be logged together so we can audit weird behaviors and confirm every turn gets meaningful affect signals.
- Expanding diagnostics for logic coverage (the "complete project sweep") is a prerequisite before trusting probes for system-wide validation.

## Next steps
1. Rework the hormone gating so each turn consults the classifier, maps the inferred charge to endocrine boosts (realistic peaks and decay), and keeps the new state internal before rerunning the manual/probe sequences.
2. Expand the diagnostics cascade into a codic-style project sweep (per the `roadmap` entry) that touches `app/`, `brain/`, `state_engine/`, `hormones/`, and `scripts/` to catch disconnected modules or logic bugs, log the grind, and gate CI on the results.
3. Execute the LoRA training/inference pipeline noted in `temp.md`: prepare the labeled dataset, convert/download Qwen weights, run `train_affect_lora.py` with eval metrics (batch 64, LR ~2e-4, rank 16-32), build the wrapper, and swap it into `brain/affect_classifier`.
4. Reevaluate the manual/probe transcripts once the above landings are in place, keep the `logs/webui/manual_transcript.txt`/`logs/probe_runs/affect_validation.json` files updated, and collect the training/eval metrics so future checks have baselines.
