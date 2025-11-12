# Session Notes – 2025-11-06

## What Happened
- Hardened the controller clamp: added a recovery/priming window that keeps self-bias negative, boosts outward tags, trims tokens, and records the active windows in telemetry so we can see when clamps fire (`main.py` updates).
- Softer reinforcement balances: authenticity now gives extra credit for leading with “you/we” and self-preoccupation penalties focus on repeated first-person leads rather than every “I”, reducing the collateral damage when we clamp (`brain/reinforcement.py`).
- Mid-span harness upgrades:
  - Each run spins up `logs/probe_runs/mid_span/harness-N` automatically.
  - Added per-run `summary_compact.json` so we can skim auth/self/drift per iteration without opening every JSONL file.
  - Live warnings/logged streak metrics print during injector runs, highlighting when a profile stays above the self-preoccupation gate for ≥5 iterations.
- Ran three diagnostic harnesses (harness-3..5) after the changes to measure whether the clamp keeps self-preoccupation below 0.75 while holding authenticity ≥0.45.

## Observations / Keep in Mind
- Harness-4 and harness-5 still show long high-self streaks (9–13 consecutive turns) even though the clamp fires; only a couple of turns (e.g. harness-5 iteration 8 and 27) simultaneously meet auth ≥0.45 and self ≤0.75.
- Authenticity averages slipped (≈0.42 instruct / 0.38 base) because the clamp bias stays negative for the whole recovery window; we saw multiple low-self turns with auth <0.3.
- Session resets triggered by the clamp briefly drop self_preoccupation (<0.2) but the very next turn rebounds if we don’t explicitly steer the response outward.
- The new `summary_compact.json` files make it easy to spot the exact iterations where the gate was met, so we can evaluate future tweaks quickly.

## Next Steps
1. **Outward priming post-reset** (now queued via `_reset_live_session`): monitor the next harness to confirm the priming event actually extends the low-self streak; adjust the priming text/logit boost if it doesn’t.
2. Tune the priming/recovery windows: if self-preoccupation still bounces back on the second turn, consider longer priming or an adaptive decay that waits for two good turns before releasing the clamp.
3. Re-check authenticity impact once priming lands—goal is auth ≥0.45 for at least three consecutive low-self turns per profile.
4. When the mid-span harness stabilises, promote the priming hook into the overnight diagnostics pipeline so the canary loop benefits too, then circle back to the “voice guard rollout” work that’s blocked on these behavioural fixes.
