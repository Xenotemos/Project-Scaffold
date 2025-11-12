# Update 0.11 - Temporal Working Memory

## Summary
- Added a duration-aware working-memory buffer that automatically expires conversations older than the configured window.
- Rebuilt the tick cycle so working memory refreshes alongside hormone decay and consolidation.
- Augmented the memory unit tests to cover the new short-term decay behaviour.

## Associated Files
- `dev-log.md`
- `memory/manager.py`
- `tests/test_memory.py`

## Validation
- `python -m unittest discover -s tests`
