# Repository Guidelines

## Project Structure & Module Organization
- `main.py` – FastAPI entrypoint orchestrating state, persona, controller policy, and LLM routing.
- `app/` – shared runtime helpers (`constants.py`, `settings.py`, `chat_context.py`) that keep heuristics/config out of `main.py`.
- `brain/` – runtime intelligence (controller policy, hormone model, reinforcement, voice guard, llama engine).
- `state_engine/`, `memory/`, `hormones/` – organism simulation primitives.
- `scripts/` – operational tooling (`ci_sanity.py`, `probes/` harnesses, training scripts).
- `docs/` – planning notes, logs, roadmaps; `docs/corpus/` stores curated transcripts.
- `tests/` – pytest suites covering controllers, guards, diagnostics.

## Build, Test, and Development Commands
- `python -m scripts.ci_sanity --mid-span-dir path/to/run` – dry-run hormone/controller training and probe gating.
- `python -m scripts.probes.affect_validation --profiles instruct base` – multi-scenario affect probe (fails on missing deltas).
- `.\.venv-win\Scripts\python.exe -m pytest` – run unit tests (use full path on Windows, or `python3 -m pytest` in WSL).
- `python main.py` – launch the FastAPI service (ensure `.env` or `config/settings*.json` is in place).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, black-like spacing but no auto-format enforced; keep imports ordered stdlib/third-party/local.
- Modules follow snake_case; classes use PascalCase; config JSON keys are lower_snake.
- Favor descriptive function names (e.g., `_apply_controller_adjustments`, `load_controller_policy`). Use inline comments only for non-obvious heuristics.

## Testing Guidelines
- Pytest is the primary framework; tests live in `tests/` with files named `test_*.py`.
- Aim to cover heuristic edges (e.g., helper voice penalties, controller token floors). When adding probes, include regression tests or probe fixtures.
- Use `pytest -k pattern` for focused suites; ensure new modules expose minimal seams for unit tests (pure functions, injectable dependencies).

## Commit & Pull Request Guidelines
- Keep commits scoped (“voice_guard: add helper penalty decay”). Reference work units in the subject if applicable.
- Pull requests should describe behavioral impacts, mention probe/test evidence, and link roadmap items or session notes. Attach logs/screenshots for UI or probe changes when relevant.

## Security & Configuration Tips
- Secrets and model paths live in `config/` or environment variables (e.g., `LLAMA_MODEL_PATH`, `LIVING_SETTINGS_FILE`); never hardcode credentials.
- Logging directories (`logs/`, `docs/logs/`) may grow quickly—rotate before long probe runs. Use `_reload_runtime_settings()` to refresh models after editing configs.

## Agent Workflow & Shell Expectations
- Keep inner monolouge and reasoning professional and focused to avoid using unnecesary context tokens.
- Default to the Codex CLI bash flow (`/mnt/c/...` paths, `ls`, `sed`, `python <<'PY'`) to mirror existing diagnostics scripts and avoid Windows path quirks.
- When Windows tooling is required (e.g., `.ps1` helpers), invoke PowerShell explicitly via `powershell.exe -Command "..."` from bash; otherwise stay in bash for consistency.
- Prefer `rg`, `python - <<'PY'`, and `pytest` via the virtualenv (`.\.venv-win\Scripts\python.exe`) to keep analysis, edits, and tests aligned with current workflows.
