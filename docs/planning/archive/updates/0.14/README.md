# Update 0.14 - Output Headroom
#
## Summary
- Increased llama.cpp completions to 768 tokens by default and exposed a settings override.
- Raised client timeouts so long generations survive the bridge end-to-end.
- Updated configuration docs to highlight the new token cap knob.
#
## Associated Files
- `dev-log.md`
- `main.py`
- `brain/local_llama_engine.py`
- `config/settings.example.json`
#
## Validation
- `python -m unittest discover -s tests`
