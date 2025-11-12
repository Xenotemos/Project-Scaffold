# Update 0.12 - Intent Routing & Dynamic Profiles

## Summary
- Introduced a heuristic intent router to classify incoming turns (emotional, analytical, narrative, reflective).
- Applied intent-specific prompt fragments and sampling overrides so llama.cpp responses adapt tone without exposing hormone metadata.
- Surfaced detected intent in API responses and refined heuristic fallbacks to acknowledge the routing choice.

## Associated Files
- `dev-log.md`
- `main.py`
- `brain/intent_router.py`
- `brain/local_llama_engine.py`
- `tests/test_chat.py`

## Validation
- `python -m unittest discover -s tests`

