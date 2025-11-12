# Update 0.13 - Reinforcement Feedback Loops

## Summary
- Added post-turn heuristics measuring valence shift, length balance, and token entropy.
- Converted reinforcement signals into hormone adjustments to subtly steer future sampling without hard prompts.
- Surfaced reinforcement diagnostics in `/chat` responses and extended unit tests for the scoring path.

## Associated Files
- `dev-log.md`
- `main.py`
- `brain/reinforcement.py`
- `tests/test_chat.py`

## Validation
- `python -m unittest discover -s tests`
