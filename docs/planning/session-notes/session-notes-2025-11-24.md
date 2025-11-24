# Session Notes – 2025-11-24

- CAHM prompt cleaned (removed stray non-ASCII tokens); parser stays hardened and only falls back to heuristics on empty payloads.
- Swapped runtime to a new `torch_head` classifier (base + adapter + head.pt) so we use the trained regression heads directly; llama.cpp kept only as fallback.
- Quick gold sample eval via torch_head (n=20): valence mse 0.1245 / corr 0.70; intimacy mse 0.214 / corr 0.33; tension mse 0.107 / corr 0.10 — big lift vs. gguf generation path.
- DirectML wheel on disk is cp312-only; current env is py3.10 so torch_directml not installed. Running on CPU; acceptable (<2s per forward). Can revisit DML with a py3.12 venv.
- Telemetry shows engine + extras; rerun `scripts/probes/affect_validation.py` after restart to confirm end-to-end scoring and gating.
- Pending: decide whether to keep Q8_0 or try Q6_K from the fresh merge if tension accuracy stays low.
