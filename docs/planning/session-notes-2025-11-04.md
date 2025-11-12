# Session Notes – 2025-11-04

## What Happened
- Instrumented the chat pipeline so every turn now logs pre/post hormone vectors, reinforcement channels, sampling parameters, and memory snapshots to `logs/endocrine_turns.jsonl`.
- Added `scripts/train_hormone_model.py` to ingest the endocrine log, fit a linear forward model (`h_{t+1} = f(h_t, reinforcement, intent, stimulus)`), and emit learned coefficients to `config/hormone_model.json`.
- Introduced `brain/hormone_model.py` plus runtime wiring in `main.py` so `_update_metric_history` and `_apply_feedback` use the learned deltas instead of purely heuristic adjustments; heuristics act as fallback when no model is available.
- Landed the recurrent controller policy: `controller_policy.json` + runtime now blend traits, hormones, and memory tags into sampling/logit adjustments, with telemetry logged each turn.
- Re-trained the hormone dynamics and controller policies against the latest `logs/endocrine_turns.jsonl` capture (24 train / 6 validation samples) and refreshed the config artefacts.
- Coupled long-term memory with endocrine traces: events persist hormone/controller metadata, selector scoring reacts to spikes, and diary reflections log structured deltas for retraining.
- Automated continuous probes via `scripts/continuous_probes.py`, logging nightly multi-profile metrics (authenticity, drift, self_preoccupation, hormones) with promotion gates in `logs/probe_runs.jsonl`.
- Rebuilt the diagnostics harness: `scripts/diagnostics.py` now emits machine-readable reports, auto-attempts repairs, and covers hormones↔memory, router↔sampling, reinforcement, and HTTP endpoints.
- Hardened llama.cpp startup with a warmup completion call to eliminate first-turn 400s during scripted benches.
- Extended `docs/planning/roadmap.md` with the detailed endocrine feedback rebuild plan (instrumentation → model → policy controller → continuous probes).

## Keep in Mind for Future Runs
- The current linear/controller models rely on 24 train / 6 validation samples; capture richer logs before trusting them for enforcement.
- `config/settings*.json` now expect `hormone_model_path`; profile switches must reload settings so `_reinitialize_hormone_model()` can pick up updated weights.
- Streaming responses still grab telemetry from `_generate_chat_reply` when the local engine falls back; ensure future streaming changes keep that handshake intact.
- Bench scripts (`python -m scripts.bench_profiles`) remain the fastest way to capture comparable stats for instruct vs. base profiles—run them after major hormone tweaks.
- Watch the first-turn warmup: if a bench still hits the 400 path, re-run after a cold start to confirm the warmup call fired; otherwise adjust the warmup delay.

## Next Steps
1. Gather a larger endocrine log from fresh chat/bench runs and retrain the hormone/controller policies; target lower cortisol/dopamine RMSE before loosening safeguards.
2. Expand reinforcement scoring to multi-dimensional outputs feeding the model (authenticity, drift, concreteness, self-preoccupation).
3. Integrate probe outcomes with deployment tooling: surface `logs/probe_runs.jsonl` dashboards and wire promotion gates into release CI.
4. Run an authenticity lift pass on probe prompts/score weights, then retrain endocrine models with the improved transcripts.
