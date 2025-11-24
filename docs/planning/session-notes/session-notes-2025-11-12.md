# Session Notes - 2025-11-12

## Highlights
- Manual chats and affect validation still play helper tone and clip at 64 tokens even after dopamine/oxytocin shifts; controller clamps and the legacy prompt still dominate the surface voice (`docs/logs/transcript.txt`, `logs/probe_runs/affect_validation.json`).
- Hormone tracing now defaults to enabled with console confirmation, and `scripts/probes/run_affect_validation.cmd` forces tracing so operators can replay the multi-scenario probe and compare `logs/probe_runs/affect_validation.json` with `logs/hormone_trace.jsonl`.
- Fixed the residual “positive” stimulus on reset, ensured affect-style overrides reapply their min-token floors after controller adjustments, and stopped the boot-time UnboundLocalError that occurred when the probe launched right after startup.
- Reworked the persona/system prompt toward body-first narration, tightened voice guard penalties, and gave the base profile an extra dopamine boost for affectionate turns so both profiles now hit the probe expectations.
- Observability plan: keep manual chat transcripts, probe reports, web UI interactions, and telemetry snapshots under `docs/logs/` to build a corpus for future fine-tuning and CI checks.
- Web UI logging now emits both the raw JSONL feed and a readable summary stream (`docs/logs/webui/interactions_readable.log`) with sampling, controller, and hormone snippets per turn; probe harness uses the same data for triage.
- Local llama warmup handling is more forgiving: repeated `400 Warmup` errors now retry inside `brain/local_llama_engine.py`, so affect-validation runs no longer fail when swapping profiles or cold-starting the base model.

## Next Steps
1. Finish clamp enforcement so affect-driven max-token floors can’t be overridden, and expand helper-tone penalties/regeneration (Roadmap vision items #6–#8).
2. Deepen the affect probe into richer multi-turn scenarios and wire it into `scripts/ci_sanity.py` so hormone deltas and voice metrics gate each change.
3. Collect embodied vs. helper transcripts for the persona fine-tune once the runtime loop delivers a lived-in voice.
4. After each manual + probe cycle, skim `docs/logs/webui/interactions_readable.log` alongside the JSON to spot helper slips quickly and stash the best transcripts for corpus building.
5. Monitor llama warmup behavior after profile swaps; if 400s persist beyond the new retries, consider preloading both models or running staggered warmups before probes.
