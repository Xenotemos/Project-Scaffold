## 0.17.1 - 2025-10-29 Emergent Reflection Loop
- Tuned reinforcement heuristics to reward self-referential authenticity, penalize assistant drift, and route the new metrics into hormone feedback for richer affect swings.
- Wove internal reflection notes into memory (tagged `internal`/`reflection`) and biased the selector so tension/curiosity resurface those narratives without scripting behaviour.
- Expanded trait-driven sampling swings and added self-observation logit biases, then captured authenticity + sampling snapshots to JSONL logs for calibration; refreshed the probe script runs to record five-cycle sweeps.


## 0.17 - 2025-10-28 Affect Debug & Calibration
- Added affect-aware sampling blend outputs to the API/UI, including an optional debug drawer that surfaces traits, tags, and live sampling parameters when enabled.
- Logged every sampling snapshot to `logs/sampling_snapshots.jsonl` for offline analysis and hooked the probe script to run multi-cycle stimuli sweeps.
- Extended the local llama engine to honor presence penalties and ensured the new configuration flags propagate into responses for downstream calibration tools.

## 0.16 - 2025-10-27 Streaming & Controls
- Replaced the heuristic intent router with a hand-tuned linear classifier that produces softmax confidences and rationale tokens.
- Added `/chat/stream` SSE endpoint plus `static/chat.js` streaming client; tokens render incrementally with JSON completion metadata.
- Expanded `tests/test_chat.py` with streaming stubs, alternating length-plan assertions, and admin endpoint coverage (caps + reload).
- Introduced runtime reload + caps endpoints (`/admin/reload`, `/admin/caps`) and refactored config loading to support hot restarts without process restarts.
- Surfaced config settings via `_refresh_settings` + `_configure_clients`, ensuring llama server startup happens only when explicitly configured.
- Shifted the local llama bridge to `/v1/chat/completions`, added stop sequences + negative role bias, softened persona/memory phrasing, and guarded against `User:/Assistant:` transcript artifacts in both heuristics and tests.

## 0.15.1 - 2025-10-27 Local Llama Boot Fix
- Added `config/settings.json` with the llama-server binary/model paths so FastAPI initializes the local engine instead of falling back.
- Logged a diagnostic in `_init_local_llama` when the configuration is missing, making future boot issues obvious in `logs/logs.txt`.
- Smoked the fix with `python -c "import asyncio, main; result = asyncio.run(main._generate_chat_reply('Quick local health check.')); print(result[1])"`; response source returned `local` and the server was torn down afterward.

## 2025-10-27 Integration Touchups
- Normalized chat UI and persona truncation to ASCII-only symbols across `main.py`, `templates/chat.html`, `static/chat.js`, and `static/chat.css`.
- Ran automated uvicorn smoke test hitting `/`, `/state`, and `/chat`; heuristic replies confirmed hormone-memory-persona linkage (`logs/transcript.txt`).

## 0.10 - 2025-10-27 Emergent Sampling Shift
- Added noradrenaline support to the hormone system and refreshed persona cues to stay descriptive without naming physiology.
- Derived llama.cpp sampling parameters (temperature, top_p, frequency_penalty, logit bias) directly from hormone deltas and passed them to the local engine instead of embedding hormone text in prompts.
- Scrubbed hormone/persona details from model contexts and updated heuristic fallback messaging to avoid hormone language while keeping conversational awareness.

## 0.11 - 2025-10-27 Temporal Working Memory
- Introduced a sliding working-memory window that retains recent turns for up to two minutes while capping the buffer size.
- Wired the decay tick to rebuild that transient buffer so stale exchanges fall out naturally.
- Extended memory tests to assert both consolidation and the new expiry behaviour.

## 0.12 - 2025-10-27 Intent Routing & Profiles
- Added a lightweight intent router that classifies turns (emotional, analytical, narrative, reflective) using contextual heuristics.
- Mapped each intent to prompt fragments and sampling overrides so llama.cpp generation shifts tone without leaking hormone text.
- Extended heuristic fallbacks, chat responses, and tests to surface the active intent and validate the new routing path.
## 0.13 - 2025-10-27 Reinforcement Feedback
- Added post-response heuristics (valence shift, length balance, lexical entropy) to score each AI reply.
- Mapped reinforcement scores into hormone adjustments so future turns self-tune without explicit tone commands.
- Surfaced reinforcement diagnostics in API responses and extended tests to cover the new scoring path.

## 0.14 - 2025-10-27 Output Headroom
- Raised the default llama.cpp completion cap to 768 tokens and wired settings/config overrides into the local engine.
- Increased client timeouts so downstream bridges allow longer generations without clipping.
- Documented the new knob in `config/settings.example.json` for consistent deployment tuning.

# Living AI Session Notes
## 0.15 - 2025-10-27 Ambient Stimuli
- Injected time-based noise and simple sentiment cues into the hormone system each tick/event.
- Refreshed heuristic replies to reference ever-changing cues instead of static tone hints.
- Added reinforcement-driven adjustments (already present) to embrace micro-fluctuations across turns.


## Summary
- Set up FastAPI service with hormone/state/memory orchestration and async ticking.
- Added REST endpoints: `/ping`, `/state`, `/memories`, `/events`, and heuristic `/chat`.
- Built browser chat UI (`templates/chat.html`, `static/chat.css`, `static/chat.js`) displaying live mood and supporting stimulus options.
- Resolved SQLModel reserved-field issue by renaming `metadata` to `attributes` in `LongTermMemoryRecord`.
- Added tests for hormones, memory, and chat heuristics; httpx-dependent tests skip if library missing.
- Installed project dependencies in WSL environment; verified Uvicorn runs via `wsl python3 -m uvicorn main:app --reload --app-dir "/mnt/c/Users/USER/Desktop/Project Scaffold"`.
- Scaffolded separate `living_llm` repo with cognitive bridge, config loader, FastAPI service, scripts, and tests (located at `/home/xeno/living_llm`).

## Next Steps
1. Replace heuristic `/chat` reply with `living_llm` bridge once the external LLM endpoint is ready.
2. Swap `datetime.utcnow()` usages for timezone-aware timestamps.
3. Install `httpx` to enable currently skipped FastAPI test suite.
4. Integrate `/chat` events with `living_llm` service and expand API coverage.

## Commands
- Start API (WSL): `wsl python3 -m uvicorn main:app --reload --app-dir "/mnt/c/Users/USER/Desktop/Project Scaffold"`
- Run tests: `python3 -m unittest discover -s tests`
- Living LLM service: `uvicorn services.run_server:app --reload` from `/home/xeno/living_llm`

Keep this file for context when resuming work.

## Latest Session – GPU Integration Prep (2025-10-22)
- FastAPI `/chat` now prefers a direct LLM bridge with heuristic fallback; timestamps are timezone-aware and tests run cleanly with `httpx`.
- Added `brain/llm_client.py` plus updated tests using `httpx.ASGITransport`; `requirements.txt` includes `httpx`.
- Verified local GGUF models under `D:\AI\LLMs` (Mistral/Nemo/GPT variants) and system specs: Ryzen 5 5600X, RX 7600 XT 16 GB, WSL2 kernel `6.6.87.2`.
- Goal: build a GPU-accelerated llama.cpp (HIP/Vulkan) on Windows, expose it to our app (binary wrapper or lightweight IPC) so `/chat` serves GPU-backed replies without SillyTavern/REST intermediaries.
- Next actions: install/compile llama.cpp with AMD backend, confirm the binary path + args, then integrate via a `LocalLlamaEngine` wrapper or provide an equivalent local API endpoint.
- Added `scripts/build_llama_cpp.py` to automate cloning and compiling llama.cpp (HIP/Vulkan targets) and new `brain/local_llama.py` runtime that starts `llama-server` and feeds `/chat`. Environment toggles (`LLAMA_SERVER_BIN`, `LLAMA_MODEL_PATH`, etc.) select the local GPU path ahead of the HTTP bridge.
- FastAPI startup now boots the local llama engine when configured; `/chat` prefers `"local"` source replies, falling back to `"llm"` (remote bridge) and `"heuristic"`. Tests `python -m unittest discover -s tests` still pass with the local engine disabled by default.
- Successfully built `llama-server.exe` via `python scripts/build_llama_cpp.py --backend hipblas`; binary located at `third_party/llama.cpp/build/bin/Release/llama-server.exe` (AMD HIP flag currently ignored upstream—consider Vulkan if HIP support lands later).
- Created environment helper scripts (`scripts/llama_env.ps1`, `scripts/llama_env.sh`) that export the GGUF (`D:\AI\LLMs\mistral-7b-instruct-v0.2.Q4_K_M.gguf`) and server paths—dot source before launching Uvicorn to activate the `"local"` pipeline (includes `LLAMA_SERVER_READY_TIMEOUT=120` + extended `LIVING_LLM_TIMEOUT`/`LLAMA_SERVER_TIMEOUT=60` so slow first replies don’t time out).
- `LocalLlamaEngine` now drains llama.cpp stdout/stderr asynchronously, mirrors lines into `logs/llama-server.log`, opens a separate PowerShell tail window so GPU activity is visible, and serializes startups with a reuse-aware lock to avoid double-spawning the binary.
- Chat UI clears and refocuses the input box after each send to prevent accidentally echoing prior responses.
- Verified end-to-end chat via `/chat` UI using the local engine—multiple back-to-back prompts stayed on `"local"` with no fallbacks. Note: the UI keeps the previous assistant message in the text box, so clear it before sending the next prompt to avoid echoing the AI.
- Successfully rebuilt `llama-server.exe` with the Vulkan backend once `VULKAN_SDK` was exported (shaders offload 33 layers to GPU). Env scripts (`ps1`/`sh`) now auto-detect the latest `C:\VulkanSDK\...\Bin` and prepend it to `PATH`. HIP/OpenCL builds still require AMD ROCm/OpenCL runtimes if we want additional fallbacks.

## Pending Follow-up — 2025-10-22
- Install AMD ROCm or the standalone OpenCL runtime if we want HIP/OpenCL builds available as GPU fallbacks.
- Reintroduce tuned launch flags (`--gpu-layers`, etc.) once the preferred GPU backend is settled and benchmarked.
- Benchmark Vulkan-backed inference (latency/VRAM usage) and capture baseline metrics for optimization work.

## 2025-10-23 Persona & UI Update — 21:56
- Added hormone-to-trait persona builder (`main.py`) with baseline-aware decay tweaks (`hormones/hormones.py`) so the AI speaks about sensations rather than numeric levels.
- Updated `brain/local_llama.py` system prompt to reference tone, signals, memory focus, and llama metrics, ensuring replies avoid raw hormone dumps.
- Refreshed terminal UI to show persona tone/guidance instead of hormone numbers (`static/chat.js`, CSS) and expanded the console layout.
- Extended `scripts/boot_system.py` with console-close handlers, `--auto-exit`, and detached/attached `.exe` builds; logged runs under `logs/clogs.md`.

## 2025-10-23 Silent Persona Behaviour — 19:46
- Reworked persona snapshot to surface internal behaviour tags/instructions without narrative strings; heuristics now respond with brief acknowledgements (`main.py`).
- Added feedback loops that nudge hormone deltas after AI responses based on persona cues, keeping future turns balanced (`main.py`).
- System prompt (`brain/local_llama.py`) now treats persona cues as hidden instructions; UI displays compact status tags via `static/chat.js`.
- Verified multi-turn exchange via detached launcher (logged in `logs/clogs.md`) to confirm the model adapts tone silently while avoiding hormone exposition.

