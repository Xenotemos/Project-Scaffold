# Update 0.15.1 - Response Length Planner
#
## Summary
- Added a response length planner with brief/concise/detailed bands that feed the system prompt and sampling overrides.
- Reinforcement signals now consider the target band to discourage runaway verbosity while rewarding balanced replies.
- API responses expose the length plan metadata alongside intent and reinforcement diagnostics.
#
## Associated Files
- `dev-log.md`
- `main.py`
- `brain/local_llama_engine.py`
- `tests/test_chat.py`
#
## Validation
- `python -m unittest discover -s tests`
