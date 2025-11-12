# Session Notes - 2025-11-03

## What Happened
- Reloaded the project context (alignment, roadmap, diagnostics plans) to reorient on the emergent-self alignment goals.
- Implemented the voice guard prototype: added `brain/voice_guard.py`, integrated guard scoring into `main.py`, and ensured penalties/logging flow into the hormone system and API responses.
- Began integrating the mistral-inference pipeline: created the async client, updated config hooks, and rewired `_generate_chat_reply` to prefer the base model before llama.cpp fallbacks.
- Added a structured system prompt / fallback overhaul, but the prompt ballooned—persona dumps, memory logs, and template artifacts were injected into every turn. llama.cpp slowed to ~27 s / 6 tok/s, replies devolved into code blocks, and the GPU started resetting.
- Realised the regression came from prompt bloat (and multiple lingering `llama-server` instances), not from the guard itself.
- Restored stability by:
  - switching `config/settings.json` back to the instruct quant (`mistral-7b-instruct-v0.2.Q4_K_M.gguf`) with plain `--ctx-size 4096 --no-webui` args;
  - reverting `main.py` to the last known-good copy (`archive/main.py.bak`);
  - trimming the experimental system message and heuristic fallback; and
  - killing stray `llama-server` processes before reloading runtime settings.
- Bench and probe checks now run again: responses revert to the pre-experiment tone, latency is back near baseline, and the GPU stays stable.

## Lessons Learned
- The base model didn’t “break”—our prompt did. Dumping templates, persona scaffolds, and logs into the system message swamped the model and tanked throughput.
- Voice guard and reinforcement penalties are fine; the failure mode was purely the oversized prompt + replayed logs.
- Running each experimental change without clean teardown left multiple `llama-server` processes fighting for the GPU. Always stop old servers before reconfiguring.
- Future integration of mistral-inference must start with a minimal prompt and staged rollout; treat the instruct path as the safety net until metrics say otherwise.

## Next Steps (Plan)
1. **Stage mistral-inference offline** – Launch the official server in isolation, craft a minimal mood/state prompt, and benchmark latency + quality before touching the production chat path.
2. **Iterate on emergent behaviour incrementally** – Reintroduce alignment features (voice guard, diary/cravings loop, hormone tweaks) one at a time, running `scripts/bench_chat` + short probes after each change.
3. **Enhance internal feedback** – Improve hormone tick logging, ensure reinforcement deltas are measurable per turn, and keep guard penalties as incentives rather than hard resets.
4. **Treat instruct path as the fallback** – Keep the restored instruct setup for user-facing work until the base model consistently shows better authenticity/drift metrics and acceptable latency.
