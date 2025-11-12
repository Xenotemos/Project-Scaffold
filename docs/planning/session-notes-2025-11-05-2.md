# Session Notes – 2025-11-05 (Evening)

## What Happened
- Reorganised the project scaffold: specs moved under `build/specs/`, the legacy `main.py.bak` archived, hormone helpers relocated to `brain/hormone_model.py`, and diagnostic wrappers now live in `scripts/`.
- Expanded `brain/reinforcement.py` with a richer somatic/introspective lexicon so authenticity scoring responds to inward language; quick canaries now spike authenticity above the 0.5 target.
- Grew the diagnostics harness:
  - Added canary probes, probe/log sentinels, and smarter repair helpers that can dry-run retraining.
  - Refactored CLI flow so the cascade streams each check with 0.5 s cadence, prints a compact summary banner, and only then runs repairs with tagged two-second pauses.
  - Introduced `scripts/ci_sanity.py` plus `--dry-run` flags on the trainers to support CI guardrails.

## Outstanding / Follow-Ups
- Canary self-preoccupation is still > 0.94; need controller/self-bias dampers and endocrine tweaks before the diagnostic cascade can auto-promote.
- Prune or regenerate low-authenticity probe logs (e.g. `probe_log_20251105_095525_0003.jsonl`) so the new log sentinel reflects current behaviour.
- Fold the affect-aware reinforcement roadmap items (richer affect capture, log persistence, runtime mapping) into the next sprint once the monitoring uplift is closed out.
- Run a few guarded `scripts.continuous_probes --iterations 1` passes after each tweak so fresh baselines drive the rolling averages and confirm the dampers land.
- Retrain hormone and controller models against the updated endocrine logs once the affect metrics settle, then spot-check probe deltas before promotion.
- Draft the voice guard rollout plan (verdict persistence + regeneration loop) so it is ready to wire in as soon as diagnostics stabilise.
