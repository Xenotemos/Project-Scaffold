# Update 0.10 - Emergent Sampling Shift

## Summary
- Moved hormone-driven behaviour from prompt text into llama.cpp sampling parameters for emergent tone control.
- Expanded the hormone model with noradrenaline and persona cues that stay descriptive without exposing physiology.
- Sanitized LLM contexts while keeping heuristic fallbacks conversational and hormone-free.

## Associated Files
- `dev-log.md#L1`
- `main.py`
- `brain/local_llama_engine.py`
- `hormones/hormones.py`
- `tests/test_chat.py`
- `roadmaps/natural-tone.md`

## Validation
- `python -m unittest discover -s tests`

## Next Action
- Revisit system prompt/metrics exposure per roadmap once the pending upgrade cascade completes.
