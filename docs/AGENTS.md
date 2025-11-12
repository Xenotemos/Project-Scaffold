# Living AI Project ‚Äì Agent Instructions
Goal: Build a modular AI framework with dynamic emotional state simulation, persistent memory, and a connected language model that serves as the cognitive brain.

---

# Prefix commands with "timeout -s KILL -k 5 60"

## Overview
This system simulates an emotionally responsive AI entity whose behavior, dialogue, and internal state evolve dynamically over time.  
It integrates:
- A **Cognitive Core (LLM)** for reasoning and conversation
- A **State Engine** driven by biological analogues (hormones, moods)
- **Persistent memory** across sessions
- **APIs** for interaction and visualization

---

# llama.cpp GPU Bridge
- Run `python scripts/build_llama_cpp.py --backend hipblas` to clone and compile `llama-server` with AMD HIP acceleration (use `--backend vulkan` or `--backend opencl` as fallbacks). Vulkan needs the SDK (`glslc`) on PATHóonce exported the build offloads layers to the GPU. OpenCL still requires AMD's runtime for that option.
- Export `LLAMA_SERVER_BIN`, `LLAMA_MODEL_PATH`, optional `LLAMA_SERVER_ARGS`, `LLAMA_SERVER_HOST`, `LLAMA_SERVER_PORT`, and `LLAMA_MODEL_ALIAS` so FastAPI boots the local llama.cpp server automatically. Helper scripts: `scripts/llama_env.ps1` (PowerShell) and `scripts/llama_env.sh` (WSL) now auto-detect the latest Vulkan SDK, prepend its `Bin` to PATH, and set `LLAMA_SERVER_READY_TIMEOUT=120`, `LIVING_LLM_TIMEOUT=60`, `LLAMA_SERVER_TIMEOUT=60`. A separate PowerShell window tails `logs/llama-server.log` once the engine comes onlineóclose or reuse it between restarts, and clear the chat textarea between turns to avoid echoing the previous assistant reply.
- With these variables set, `/chat` uses the `"local"` GPU path by default and falls back to the remote bridge or heuristic replies when unavailable.

---

## Components

### 1. **Memory System**
Handles short, working, and long-term memory with decay and reinforcement.

**Modules**
- `ShortTermMemory`: temporary (context buffer, expires in minutes)
- `WorkingMemory`: currently active context and relevant facts
- `LongTermMemory`: SQLite persistence; stores key events, relationships, emotions, and summaries.

**Features**
- Automatic summarization and pruning
- Emotional tagging (link memories to hormone levels at time of storage)
- Periodic consolidation: promotes strong memories to long-term

---

### 2. **Hormone System**
Simulates biochemical drives that shape behavior and tone.

**Variables**
- Dopamine (reward/motivation)
- Serotonin (stability/calm)
- Cortisol (stress/alertness)
- Oxytocin (bonding/affection)

**Mechanics**
  - Exponential decay per tick (time-based)
  - Stimulus mapping: each event or user interaction modifies hormone levels
  - Feedback loops: e.g. high cortisol dampens dopamine effect
  - Output is a normalized emotional vector used by the Cognitive Core to adjust tone.
  - Baseline anchoring: decay eases toward temperament setpoints (dopamine/serotonin ò50, cortisol ò30, oxytocin ò40) instead of collapsing to zero.

---

### 3. **State Engine**
Acts as the body‚Äôs ‚Äúnervous system.‚Äù

**Responsibilities**
- Tick loop (async)
- Reads hormone values ‚Üí derives `mood_state` (happy, calm, stressed, affectionate, etc.)
- Updates memory context based on interaction history
- Broadcasts current emotional + cognitive state to all modules (API/UI/LLM)

**Output JSON Example**
```json
{
  "mood": "focused",
  "hormones": {"dopamine": 72, "serotonin": 65, "cortisol": 22, "oxytocin": 48},
  "memory_summary": "recent positive interactions",
  "timestamp": "2025-10-20T14:32:00Z"
}
```

---

### 4. **Persona & Tone Layer (2025-10-23)**
  - `main.py` generates a persona snapshot each turn: classifies hormone deltas, maps them to tone/guidance, and blends in memory summary + focus so replies describe felt experiences.
  - Persona is threaded through `/state`, `/chat`, and the LLM system prompt; the UI header now reflects tone/guidance instead of raw hormone numbers.
  - `brain/local_llama.py` uses the persona data (plus llama metrics) to instruct the model to speak in the first person, avoid numeric hormone readouts, and ground GPU commentary in supplied telemetry.

---

## Boot & Test Workflow
- Default launcher: `dist\living_ai_boot_detached.exe --no-browser --warmup-message "<note>"` (detaches after warmup for unattended runs).
- Live logs when needed: `dist\living_ai_boot_attached.exe` (press ENTER or use `--auto-exit N`; console-close handler terminates uvicorn + llama-server).
- Both executables rely on `scripts/boot_system.py` / `boot_attached.py`, which support warmup chat, browser suppression, and cleaned shutdowns.
- After manual runs, log outcomes in `logs/clogs.md` and ensure no stray `uvicorn`/`llama-server` processes remain.
