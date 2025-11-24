# Session Notes - 2025-11-19

## Summary
- Finished preparing the 5.8k-row affect dataset (`affect_dataset_run002_plus_goe.jsonl`) by merging the latest 3.8k probe harvest with 2k GoEmotions rows (capped per label) and added ASCII-normalized, deduped exports.
- Re-ran the LoRA trainer with LR=3e-4, `grad_acc=8`, and 2 epochs; final eval loss settled around **0.5200**. Added trainer logging (`LOTRAINER_LOG`) so every run is captured automatically.
- Integrated the LoRA affect head into `brain/affect_classifier` with lazy loading, and swapped `config/affect_classifier.json` to point at the adapter. Manual chats + probes now exercise the LoRA classifier, but latency jumped (~10s/turn) because PyTorch is running the full 1.7B model on CPU.
- Began work on making the affect head lightweight: merged the adapter into a HF checkpoint (`affect_lora_merged`) and attempted HF→GGUF conversion + quantization. Tried both `convert_hf_to_gguf.py` and `convert_lora_to_gguf.py`; the former never produced a GGUF, while the latter emits an adapter-only GGUF that can't be quantized (errors with `qwen3.context_length`). Built `llama-quantize.exe`, but it couldn't proceed until a full-model GGUF existed.

### Evening Deployment Recap
- Converted the merged Qwen3-1.7B affect head to GGUF (F16 + Q6_K) and quantized it via `llama-quantize.exe`, then stood up a dedicated llama.cpp sidecar (Concurrent Affect Head Module / CAHM) on port 8082. FastAPI startup now launches and health-checks the sidecar so the affect head stays hot.
- Rewired `brain/affect_classifier.py` to call the CAHM service (fast readiness probe only), capture per-call latency, and stash the model's reasoning text for downstream telemetry.
- Added CAHM telemetry sinks: a raw feed (`logs/affect_head_raw.jsonl`) plus a compact readable stream (`logs/affect_head_readable.log`). Replaced the PowerShell tail with a Python console (`scripts.affect_head_monitor`) that redraws a two-section view (scores/emotion summary above, reasoning below) so the CAHM panel mirrors the hormone/behaviour telemetry style.
- Extended `scripts/diagnostics.py` with affect-sidecar readiness + telemetry freshness checks so CI/diagnostics fail fast if the CAHM service or log tail goes stale.
- Updated the roadmap (affect-aware rollout section) to mark the quantized sidecar + telemetry work complete.
- Hardened concurrency: `StateEngine.tick()` and `register_event()` now share an `asyncio.Lock`, and the FastAPI/chat flows were updated to await the async API so background decay can’t race user-facing hormone updates.
- Living LLM client gained retries with exponential backoff for transient HTTP/network failures, reducing brittle downstream outages.
- Requirements are now pinned (`fastapi==0.119.1`, etc.) to make installs reproducible; noted this alongside the CAHM deployment to keep infra state in sync with the new dependency expectations.
- Captured a dedicated “CAHM Reliability & Dataset Plan” outlining the next push: richer archetype slices, uncoupled labels, new prompt with reasoning output, and validation steps to keep valence/intimacy/tension independent.
- Harvest + rescoring: generated a 3.7k archetype-balanced subset, then auto-scored every sample via a local LLM (`run003_external_rescored.jsonl`) so valence/intimacy/tension/confidence vary per utterance (not just per archetype). Next we spot-check batches, dedupe if needed, and merge into run003 before retraining the LoRA head.

## Keep in Mind
- Diagnostics still triggers the first LoRA load (lazy-load helps, but GPU-less runs remain slow). Until we deploy a lightweight inference path (quantized GGUF + llama.cpp server or a smaller base model), manual chats/probes will remain sluggish.
- We still owe an async soft-hold window so CAHM can return late without blocking the current turn (today it's strictly synchronous). 
- Canary authenticity (instruct) and the legacy probe log are still below gate; need refreshed baselines with CAHM live.

## Next Steps
1. **Async blend window**: add a configurable soft deadline (e.g., 500–700 ms) when waiting for CAHM so current turns use the sidecar when fast and gracefully fall back when slow, logging “late affect” events.
2. **Telemetry polish**: expose a summarized CAHM snapshot via `/telemetry/snapshot` (last scores, rolling latency, recent reasoning) so the live telemetry panel mirrors the hormone/behaviour sections.
3. **Probe validation**: rerun affect-validation + mid-span probes with the CAHM sidecar to capture latency/behaviour improvements and refresh the WARNing canary/probe logs. 
