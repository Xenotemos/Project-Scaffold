# Repository Guidelines

## Project Structure & Module Organization
`main.py` hosts the FastAPI entrypoint and telemetry loop. Runtime orchestration stays in `app/` (chat context, persona, runtime profiles) while adaptive reasoning, hormone dynamics, and policy loaders live in `brain/`. Persistent affect and trait state is kept under `state_engine/`, automation/probes under `scripts/`, UI artifacts inside `templates/` and `static/`, and archived telemetry plus planning notes beneath `docs/`, `logs/`, and `memory/`. Tests inside `tests/` mirror the runtime packages so every module has a corresponding `test_*.py`, and generated assets remain ignored via `.gitignore`.

## Build, Test, and Development Commands
- `python -m venv .venv-win; .\.venv-win\Scripts\activate` (or `.venv/bin/activate`) creates an isolated toolchain before touching dependencies.
- `pip install -r requirements.txt` refreshes the FastAPI/SQLModel stack; add `pip install -r req.txt` only if you truly need the local secrets stub.
- `uvicorn main:app --reload --host 0.0.0.0 --port 8000` boots the service, serving `templates/` and `static/` while streaming telemetry.
- `python -m scripts.diagnostics --profile instruct` or `scripts\run_mid_span.cmd` runs the probe suite exercised in CI; pair with `pytest -q` for unit coverage.

## Shell Usage (WSL + PowerShell)
- Favor WSL bash for fast file introspection and search: `ls -lah`, `tree -L 2 /mnt/c/Users/USER/Desktop/Project\ Scaffold`, and `grep -R "TraitSnapshot" app state_engine` respect UNIX paths and run faster tooling like `rg` when available.
- Switch to PowerShell for Windows-specific scripts or tools: activating `.venv-win`, running `.cmd` probes (`scripts\run_mid_span.cmd`), editing files with `Set-Content`, or invoking the default `python` install.
- When documenting commands, include both path forms (e.g., `/mnt/c/Users/USER/Desktop/Project Scaffold` and `C:\Users\USER\Desktop\Project Scaffold`) so teammates can copy/paste into their preferred shell.
- Keep environments independent: WSL virtualenvs live under `/mnt/c/.../.venv`, while PowerShell sessions rely on `.venv-win\Scripts\activate`. Avoid mixing binaries between shells to prevent path collisions.

## Coding Style & Naming Conventions
Target Python 3.11+, four-space indents, and fully annotated functions like those in `app/controller.py`. Modules and functions stay snake_case, classes PascalCase, and exported constants come from `app/constants.py`. Derive policies through helpers such as `load_controller_policy` instead of instantiating globals, and funnel logging through `app.telemetry` utilities to keep JSONL output uniform.

## Testing Guidelines
Use Pytest exclusively. Run `pytest -q` before every PR and focus on impacted files (`pytest tests/test_chat.py -k outward_release`). Whenever you touch hormones, controllers, or trait snapshots, add fixtures that assert log entries in `docs/logs/` or `tmp_pytest.txt` so reviewers can replay failures. Keep integration diagnostics under `scripts/` and capture their stdout in the PR body.

## Commit & Pull Request Guidelines
Keep commit subjects imperative and scope-first, e.g., `brain: tune reinforcement decay`. Bodies should describe motivation, mention any touched config (`config/*.json`, `app/settings`), and list verification commands. PRs must link relevant notes under `docs/planning/`, summarize behavioral changes, attach UI screenshots for template/static edits, and call out schema shifts so deployers can refresh caches.

## Security & Configuration Tips
Profiles resolve through `app.config` into JSON under `config/`; never commit ad-hoc secrets there. The placeholder PAT in `req.txt` is example-only—set real tokens via `$env:TOKEN` or `.env` overrides before invoking scripts. Ensure generated logs in `logs/`, `docs/logs/`, and `docs/corpus/*.jsonl` stay gitignored, and always sanitize telemetry via `app.telemetry.log_json_line` before shipping data off-box.
