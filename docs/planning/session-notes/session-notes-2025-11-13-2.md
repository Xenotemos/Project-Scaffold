# Session Notes - 2025-11-13 (Follow-Up)

## Completed
- `brain/reinforcement.py` now exposes `ReinforcementTracker`; `app/runtime.py` owns the tracker so `score_response(..., tracker=...)` stops mutating module globals, and `scripts/diagnostics.py`, `tests/test_reinforcement.py`, and `tests/test_chat.py` were updated to build/reset trackers deterministically.
- All controller helpers moved into `app/controller.py`, `_prepare_chat_request` now passes explicit dependencies, and controller adjustments can trigger `_reset_live_session` through callbacks so `main.py` shrank and controller state now resets cleanly.
- Gemini follow-ups landed: `app/chat_context.py` now integrates `memory.selector` spotlight keys into `RuntimeState.memory_spotlight_keys`, `hormones/hormones.py` exports the shared `classify_mood` helper used by `state_engine/engine.py`, and the new `app/config.py` centralizes profile/settings resolution for both runtime and `/admin/model` handlers; the roadmap (`docs/planning/roadmap.md`) records what remains.
- Persona and telemetry responsibilities finally left `main.py`: `app/persona.py` builds persona snapshots, applies feedback, and logs reflections, while `app/telemetry.py` owns JSONL writers; shared constants (e.g., `HORMONE_FEELING_NAMES`) live in `app/constants.py`, and diagnostics now reads the active profile via `app.config.current_profile()`.
- Validation stayed green: `.venv-win\Scripts\python.exe -m pytest tests/test_voice_guard.py tests/test_controller_policy.py tests/test_reinforcement.py tests/test_chat.py tests/test_hormones.py` and `.venv-win\Scripts\python.exe scripts/diagnostics.py` both complete (diagnostics still emits the long-standing authenticity WARNs).

## Things to Keep in Mind
- Diagnostics still report authenticity WARNs; the controller/self-bias clamps or probe baselines need another pass before CI can treat the runs as clean.
- `main.py` still hosts some persona heuristics, event logging hooks, and telemetry snapshotters, and the long-term memory scorer continues to rely on magic-number weights with no pruning of stale entries.
- Automated coverage only exercises the narrow pytest subset above; none of the new persona/telemetry/config modules have dedicated tests yet.
- Tooling: WSL bash worked for quick `ls`/`sed` pipelines but it lacks `rg`, so heavy code searches still need to happen from PowerShell.
- Closed-loop probes now run inside `scripts/ci_sanity.py`, yet the roadmap item stays open until we parse the affect-validation JSON and gate on explicit hormone/style deltas.

## Next Steps
1. Finish trimming `main.py` by moving the remaining persona-reply heuristics, event logging helpers, and telemetry snapshotters into `app/persona.py` and `app/telemetry.py`.
2. Investigate and resolve the persistent authenticity WARNs by re-tuning controller/self-bias adjustments or refreshing the probe baselines.
3. Broaden automated tests to cover the extracted modules and consider running a fuller pytest suite (plus diagnostics) after each major refactor.
4. Follow the Gemini backlog by improving long-term memory hygiene (prune/sort records and replace scoring magic numbers) so the selector work keeps its signal.
5. Finish the “closed-loop probes & CI” roadmap step by parsing the affect-validation JSON output and failing CI when hormone/style thresholds aren’t met.
