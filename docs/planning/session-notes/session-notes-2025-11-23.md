# Session Notes - 2025-11-23

## What we did
- Merged guardrail_v6 dataset with bridge + legacy rows (3,743) and trained LoRA guardrail_v3; merged + quantized to qwen3_affect_guardrail_v3-Q6_K and updated affect_classifier.json.
- Added recent-turn tracking + extended affect context (last 6-8 turns, recency-aware) so CAHM sees more than single-turn text before scoring hormones.
- Updated roadmap with completed CAHM fine-tune and new follow-ups (longer context, subtle boundary data, rationale handling).
- **2025-11-24**: Decorrelated + merged 600 subtle-boundary labels; cleaned/extended training set (`train_ready_guardrail_v6_plus_subtle.cleaned.jsonl`). Trained `affect_lora_guardrail_v4` (epoch3 eval MSE 0.1008), merged & quantized to `affect_lora_guardrail_v4_epoch3_merged-Q6_K.gguf` (now stored in `D:/AI/LLMs`).

## Keep in mind
- Need to curate 300-600 subtle boundary/consent cases with indirect wording and add them to the guardrail set. (600 added; more optional but not blocking.)
- Add explicit recency decay weighting + explicit contextual snippets (unsafe chains, reversals) to affect prompt once the larger context is stable.
- Keep rationales as auxiliary loss only; lower their weight and avoid surfacing them in user logs.
