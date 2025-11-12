# Session Notes – 2025-11-07

## What Happened
- Tightened the controller clamp: added a low-self fallback release that relaxes max tokens/temperature/top-p once two consecutive turns fall ≤0.60 self_preoccupation, and instrumented priming telemetry so we can see each clamp cycle’s bias spike (`main.py` refactors).
- Priming now replays the user’s last phrase (“you mentioned …”) after every clamp-triggered reset, using cached prompts + a phrase extractor so the first post-reset reply opens outward.
- Reinforcement heuristics gained an outward-streak score plus a reset hook so we can track consecutive you/we openings across turns; tests updated to cover the new metric.
- Mid-span harness tooling: summary rows now include outward streak, the injector prints the requested runtime/deadline, the shared prompts were rewritten to force user-referencing openings, and an “auto tweak” path extends clamp/priming windows if no profile passes both gates.
- Launcher fix: sanitized the interactive “duration minutes” prompt (regex-strips the first numeric token) so entering “30” actually yields a 30-minute run instead of silently falling back to 20.
- Ran harness-8..12 (all ~8 minutes yet) to sanity-check the new controls; verified the crash from harness-11 was the `main` naming conflict and patched the injector to import the app module as `app_main`.

## Observations / Keep in Mind
- Every post-fix harness still ended after ~8 minutes because the launcher kept feeding “20”; we need a full 30-minute trace (≈30 iterations) before we can judge authenticity recovery or outward streak behaviour.
- Self-preoccupation is under control (≤0.75 almost everywhere, average ~0.52–0.69), but authenticity collapsed (0.24–0.39 averages, zero dual gate hits in harness-12) because sampling stays clamped to short max_tokens until the release logic sees outward streaks >= 0.25.
- The outward streak metric is stuck at 0.0 for all recorded turns so far; either the prompts still aren’t producing explicit user references, or the priming payload needs more concrete slotting to echo the user’s words.
- Priming bias spikes now log in `priming_trace`, but we haven’t inspected them in telemetry yet—worth checking once a long harness runs to ensure the cached phrase flows through.

## Next Steps
1. Re-run the mid-span harness via `python -m scripts.probes.mid_span_probes --duration-minutes 30 ...` (or re-launch through the fixed `.cmd`) to finally capture a full 30-minute window and verify the clamp decay over 30+ iterations.
2. Inspect the resulting `summary_compact.json` for non-zero `outward_streak` values; if they’re still zero, consider boosting the priming template with an explicit `<user phrase>` slot or raising the outward token weights.
3. Once authenticity crosses the gate on multiple consecutive low-self turns, promote the clamp/priming changes into the overnight diagnostics pipeline so canary loops inherit the stability.
