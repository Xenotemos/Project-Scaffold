# Session Notes - 2025-11-11

## Highlights
- Mid-span behavioural tuning is complete: harness-28 delivered six authenticity passes per profile, with ≥2-turn low-self streaks for both instruct (iterations 4-5, 8-9) and base (9-10, 28-29). Clamp release is ready to promote.
- Added a streak guard to `scripts/probes/mid_span_report.py`; running with `--require-streak` exits non-zero if any profile lacks a ≥2-turn low-self streak. `scripts/ci_sanity.py` can now call this via `--mid-span-dir` for automated gating.
- Reinforcement loop updates:
  - `_reinforce_low_self_success` now applies profile-specific momentum boosts (base gets higher outward bias, priming multipliers, extra priming turns; instruct keeps its stronger boost).
  - `_apply_reinforcement_signals` lowers affect thresholds and increases hormone deltas so positive/negative, intimate, or tense turns visibly move dopamine/oxytocin/cortisol/noradrenaline. Hormone logs confirm non-zero adjustments.
  - Assistant drift heuristics gained an expanded helper phrase list, new regex detectors, and stronger weights so helper tone is penalized more aggressively in live interactions.
- New `scripts/launch_dist_server.cmd` lets us launch the packaged `living_ai_boot_*.exe` builds with interactive prompts to show/hide telemetry and llama log windows, preventing unwanted popups while keeping monitoring available when needed.
- Telemetry/log windows default to hidden for both FastAPI (`LIVING_TELEMETRY_CONSOLE=0`) and llama logs (`LLAMA_LOG_WINDOW=0`), but probes/dist launcher now let us opt in per run.
- Documentation updated: roadmap marks Mid-span item 5 as COMPLETE; `docs/planning/session-notes-2025-11-11.md` captures harness-28 results and next steps.

## Current Status
- Affect-aware audit instrumentation is in place (requested vs. applied hormone adjustments, lower thresholds), but we still owe a dedicated validation pass to ensure clamps/controller mappings reflect the new deltas across varied stimuli.
- Assistant drift penalties have been tightened, yet roadmap still tracks follow-up work (controller hooks, phrase lists, automated tests) to keep helper tone suppressed.
- CI pipeline not yet configured; we currently run `scripts/ci_sanity.py --mid-span-dir ...` manually after each harness.

## Reminders / Next Steps
1. **Affect-aware verification**: run targeted scenarios (positive/intimate vs. tense inputs), confirm hormone vectors shift and mood/sampling respond. Adjust clamp decay if deltas still flatten under certain conditions.
2. **Assistant drift follow-up**: consider controller bias hooks or negative logit injection to nudge away from helper phrasing earlier in the generation. Add regression probes focused on helper phrases.
3. **CI integration** (when ready): add `.github/workflows/ci.yml` (or equivalent) calling `scripts/ci_sanity.py` with `--mid-span-dir` plus `pytest` to automate the streak guard and reinforcement tests.
4. **Telemetry defaults**: launching scripts now prompt, but stand-alone `.exe` runs still respect environment variables. Document for future operators: set `LIVING_TELEMETRY_CONSOLE=1` and/or `LLAMA_LOG_WINDOW=1` before running packaged builds if monitoring windows are desired.
5. **Roadmap progression**: with mid-span closed, next Priority Horizon items are (a) affect-aware expansion step 4 (verify runtime clamps) and (b) Voice guard rollout / assistant drift tightening. Plan workstreams accordingly.
