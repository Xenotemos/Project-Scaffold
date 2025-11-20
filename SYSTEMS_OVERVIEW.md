# Systems & Subsystems (Priority-Ordered)

## Core Runtime Path (highest criticality)
- **FastAPI entrypoint** (`main.py`) – boots the app, wiring state engine, persona, controller policy, sampling, voice guard, telemetry, and routes (`/chat`, `/state`, etc.).
- **State engine** (`state_engine/engine.py`) – orchestrates hormones, traits, memory manager; ticks endocrine + controller hooks per turn; exposes snapshots for telemetry.
- **Hormone model** (`hormones/`, `brain/hormone_model.py`, `config/hormone_model.json`) – learned/endocrine dynamics, clamps, decay, and hormone-to-style mapping (`config/hormone_style_map.json`).
- **Controller policy** (`brain/controller_policy.py`, `config/controller_policy.json`) – sampling nudges (temperature/top_p/max_tokens/bias) derived from traits/hormones/memory tags.
- **Affect classifier (runtime hook)** (`brain/affect_classifier.py`) – current rule-based affect tags; future path to LoRA adapter/inference wrapper.
- **Sampling pipeline** (`app/sampling.py`, `brain/local_llama_engine.py`, `brain/mistral_inference.py`) – intent-aware sampling knobs, helper-tone penalties, persona blending, logit biases, length planning.
- **Persona & chat context** (`app/persona.py`, `app/chat_context.py`) – builds persona snapshot, self-narration, memory previews, affect context injection.

## Embodiment & Safety Layers
- **Voice Guard** (`brain/voice_guard.py`, penalties in `app/sampling.py`) – helper-tone penalty boosts, regen triggers, verdict logging to `logs/voice_guard.jsonl`.
- **Reinforcement & hormones coupling** (`brain/reinforcement.py`, `app/telemetry.py`) – converts affect/reinforcement scores into hormone deltas; logs endocrine turns and controller traces.
- **Intent router** (`brain/intent_router.py`) – classifies incoming turns for sampling/policy hints.

## Memory & Context
- **Memory manager** (`memory/manager.py`, `memory/repository.py`, `memory/selector.py`) – short/working/long-term buffers, decay, consolidation, spotlight selection for context.
- **Chat context composer** (`app/chat_context.py`) – assembles memory previews, affect previews, self-note for each turn.

## Probes, Diagnostics, Tooling
- **Probes** (`scripts/probes/`) – affect validation, mid-span probes, overnight harnesses, affect data probe (`affect_data_probe.py`) with scenario files and per-run logs.
- **Diagnostics & CI sanity** (`scripts/diagnostics.py`, `scripts/ci_sanity.py`, `scripts/continuous_probes.py`) – health checks, gating, telemetry summaries.
- **Training scripts**:
  - Affect LoRA head: `fine_tune/train_affect_lora.py`, datasets in `fine_tune/affect_dataset*.jsonl`.
  - Hormone/controller policies: `scripts/train_hormone_model.py`, `scripts/train_controller_policy.py`.
  - Affect data exporter/inspector: `scripts/probes/export_affect_dataset.py`, `scripts/probes/inspect_affect_dataset.py`, scenario JSONs under `scripts/probes/scenarios/`.
- **Environment helpers** (`scripts/llama_env.sh/.ps1`, `scripts/build_llama_cpp.py`) – llama.cpp build, local engine configuration.

## Config & Assets
- **Config** (`config/`) – settings files (`settings.json`, `settings.base.json`), hormone/controller policies, style map, inference settings.
- **Logs** (`logs/`, `docs/logs/`) – endocrine turns, telemetry, probe runs, web UI transcripts.
- **Docs & planning** (`docs/planning/roadmap.md`, session notes, dev-log) – roadmap and session history.

## Tests
- **Pytest suites** (`tests/`) – state engine affect tests, reinforcement tests, integration flags for affect/memory previews, etc.

## Supporting Modules
- **Shared app helpers** (`app/`: `settings.py`, `constants.py`, `chat_context.py`, `runtime.py`, `controller.py`, `telemetry.py`).
- **LLM clients & routing** (`brain/llm_client.py`, `brain/local_llama.py`, `brain/local_llama_engine.py`, `brain/mistral_inference.py`).
- **Utilities** (`utils/`, `third_party/`) – settings loader, ggml/gguf tools, build scripts.

## Lower-Criticality / Misc
- **Front-end/static assets** (`static/`, `templates/`).
- **Archival/experimental** (`archive/`, `dist/`, `build/`, `gemini_cli/`, `codex/`, `attempt.md`, `temp.md`).

---
Use this as a quick map: top blocks are the runtime “red path” (requests → hormones/memory → sampling → reply); middle blocks are safety/diagnostics; lower blocks are tooling and docs. Adjust priority as the project focus shifts. 
