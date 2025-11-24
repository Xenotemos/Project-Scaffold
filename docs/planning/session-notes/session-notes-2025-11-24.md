# Session Notes – 2025-11-24

- CAHM prompt cleaned (removed stray non-ASCII tokens); parser stays hardened and only falls back to heuristics on empty payloads.
- Swapped runtime to a new `torch_head` classifier (base + adapter + head.pt) so we use the trained regression heads directly; llama.cpp kept only as fallback.
- Quick gold sample eval via torch_head (n=20): valence mse 0.1245 / corr 0.70; intimacy mse 0.214 / corr 0.33; tension mse 0.107 / corr 0.10 — big lift vs. gguf generation path.
- Built a py3.12 `.venv-dml` with `torch_directml` (torch 2.4.1) + transformers/peft; torch-head runs FP16 on DirectML (`device=privateuseone:0`), ~0.8–0.9s per classify after warmup. Keep using `.venv-dml\\Scripts\\python` to launch the app.
- Telemetry shows engine + extras; rerun `scripts/probes/affect_validation.py` after restart to confirm end-to-end scoring and gating.
- Pending: decide whether to keep Q8_0 or try Q6_K from the fresh merge if tension accuracy stays low.
