# Roadmap Cascade

## Gemini Report Follow-Ups
- [ ] TODO (Gemini) Carve the remaining persona/memory/event orchestration out of `main.py`, keeping the entrypoint focused on FastAPI wiring and top-level coordination.
- [x] Reconnect the `memory.selector` heuristics to request context by promoting spotlighted memories + previews into the chat context and runtime state.
- [x] Deduplicate `_derive_mood` vs. `hormones.hormones.derive_mood` so there is a single source of truth for endocrine-to-mood mapping.
- [x] Formalize configuration loading (centralize profile + settings resolution via `app.config`) to shrink the ad-hoc env/JSON mixing that gemini flagged.

## Priority Horizon (v0.17.x)
- [ ] **Voice guard rollout**: replace persona prose with state surfaces, wire penalty and regeneration loops for helper phrasing, and persist guard verdicts to `logs/voice_guard.jsonl`.
- **Affect-aware embodiment rollout**:
  1. [x] Automatic affect tagging – classifier runs on every user turn, writes to `logs/affect_classifier.jsonl`, and blends confidence-weighted scores into `_apply_reinforcement_signals`.
  2. [x] Runtime clamp validation – hormone trace logging + `scripts/probes/affect_validation.py` expose requested/applied deltas and act as the gating harness.
  3. [x] Hormone-to-decoder map – runtime reads `config/hormone_style_map.json`, applies per-hormone sampling overrides, and logs the bands hit each turn.
  4. [x] Behavioral response surfaces – affect-driven sampling overrides loosen or tighten cadence (max tokens, temperature, bias scales) for affectionate vs. tense turns.
  5. [x] Self-narration channel – hormone traces roll into a persistent internal note injected into persona/memory and heuristic replies so the assistant references felt state, not just metrics.
  6. [x] Assistant drift gating – helper-tone penalties now apply inside the sampler (`app/sampling.apply_helper_tone_bias`), VoiceGuard regenerations boost the penalty scale on every flagged turn, and we persist each verdict in `logs/voice_guard.jsonl` plus the telemetry snapshots.
  7. [ ] Closed-loop probes & CI – extend `scripts/ci_sanity.py` (or a sibling task) to parse the affect-validation JSON summaries, assert that hormone deltas cross configured thresholds, and diff style metrics so regressions fail fast in automation; probes already run in CI, but they do not yet enforce numeric gates.
- [ ] **Persona fine-tuning**: once the instrumentation is trustworthy, harvest transcripts that exhibit the desired entity voice, curate counter-examples of helper tone, and fine-tune an adapter/LoRA head so the base model internalizes the affect-aware style instead of relying solely on runtime clamps.
- [ ] **System prompt hygiene**: adjust `brain/local_llama_engine._build_system_message` (and sampling penalties) so guardrail strings like “Opening sentence must…” cannot be echoed verbatim, keeping embodied replies from narrating the instructions back to the user.
- [ ] **Affect classifier LoRA head**:
    - [ ] Data: curate `affect_dataset.jsonl` entries that include raw text plus numeric valence/intimacy/tension targets plus categorical tags (affectionate, tense, neutral), blending distantly labeled corpora (GoEmotions, EmpatheticDialogues) with the most recent transcripts so the classifier sees the same tone spectrum as the probes.
    - [ ] Model & runtime: convert `Qwen3-1.7B-Q8_0.gguf` (or a nearby HF checkpoint) into FP16/BitsAndBytes 4/8-bit PyTorch weights, keep the core quantized checkpoint on `D:\\AI\\LLMs`, then train rank=16-32 LoRA adapters (alpha=16, dropout ≈0.05) so the Qwen inference stack fits inside the 16 GB RX 7600XT budget.
    - [ ] Training stack: drive `train_affect_lora.py` through Hugging Face Transformers + PEFT (or Axolotl/OpenDelta wrappers) with LR ~2e-4, batch 64 (gradient accumulation as needed), 1-3 epochs, and log validation metrics while saving adapters per run under `affect_lora/`.
    - [x] Inference API: wrap the quantized checkpoint + adapter in a lightweight Python module (or local REST hook) that ingests a string and emits an `AffectClassification` record (valence/intimacy/tension/confidence/tags); persist every prediction to `logs/affect_classifier.jsonl` for auditability of false positives/negatives.
    - [x] Integration & fallback: point `brain/affect_classifier.AffectClassifier` at the wrapper while retaining the dataclass interface so downstream code stays untouched; if the adapter times out, drops below a configurable confidence floor, or the service is unreachable, fall back to the current heuristics.
    - [x] Monitoring & diagnostics: add diagnostics that ping the wrapper before chat traffic, expose the classifier output alongside hormone traces in `scripts/probes/affect_validation.py`, and gate CI on that sanity check so regressions trigger alerts early.
    - [ ] Next actions:
       1. Prepare the dataset (label transcripts, align numeric valence/intimacy/tension, and augment with external corpora).
       2. Convert/download the FP16/quantized Qwen3-1.7B checkpoint so training scripts can load it.
       3. Spin up the LoRA fine-tune with the configured PEFT stack.
       4. Harvest a few thousand labelled user turns via manual chats + probes, funnel them through a builder script into `fine_tune/affect_dataset.jsonl`, then rerun the LoRA trainer before swapping the adapter into the affect head.
       5. Expose the inference function and wire it into `AffectClassifier`.
       6. Add diagnostics/CI coverage to ensure the classifier service stays reachable.
 7. Quantize the affect head (merge + GGUF Q6_K, llama.cpp service) so classification latency drops under 2s; document fallback to heuristics when the service is unavailable. **Done**: Q6_K sidecar live; telemetry wired; async blending TBD (latency already improved). Added a dedicated CAHM monitor console (two-section redraw) to replace the noisy tail. Concurrency guard (async lock around `StateEngine.tick/register_event`) keeps CAHM hormone deltas accurate even under load.
- [x] **Mid-span behavioural fine-tuning**:
  1. [x] Teach `scripts/probes/mid_span_probes.py` to emit per-run `run_meta.json` alongside `summary_compact.json` so the full 30-minute window, iteration counts, and gating deltas are auditable without spelunking every log.
  2. [x] Ship `scripts/probes/mid_span_report.py` for quick ASCII trend reviews (authenticity/self/outward) and recent-iteration tables, making probe sign-off possible in a single command.
  3. [x] Rework outward-streak scoring so early “you/we” openings and cached priming phrases register (suffix normalization, filler-skipping, relaxed context overlap).
  4. [x] Boost authenticity scoring when outward streaks fire: scale down the self-pronoun penalty, add relational bonuses tied to outward streak + attunement, and gate an extra lift when self_preoccupation <= 0.65.
  5. [x] Keep running mid-span probes until instruct/base each hit the auth ≥0.45 & self ≤0.75 gate on consecutive low-self turns so the clamp release can be promoted into the broader v0.17.4 flow.

- [x] **Affect-aware reinforcement expansion**:
  1. [x] Enrich reinforcement scoring to spot affectionate/diminutive language, broader sentiment cues, and tension/playfulness in both user input and replies.
  2. [x] Persist the added affect metrics (valence, intimacy, tension) alongside existing reinforcement fields in `logs/endocrine_turns.jsonl` so modelling captures how prompts land.
  3. [x] Update `_apply_reinforcement_signals` to translate those affect scores into dopamine/oxytocin lifts and cortisol/noradrenaline shifts, letting high-intimacy prompts move internal state.
 4. [x] Verify that the runtime clamp still applies those hormone deltas — CAHM scores now dominate `_apply_reinforcement_signals`, and the new `affect_head_alignment.jsonl` log confirms dopamine/oxytocin adjustments match classifier reasoning.

- [x] **Diagnostics & monitoring uplift**:
  1. [x] Extend `scripts/diagnostics.py` with canary probes that hit a single instruct/base turn and assert authenticity/self-preoccupation thresholds, ensuring behavioural regressions flag red immediately.
  2. [x] Add automated log sentinels: parse `logs/probe_runs/*.jsonl` and `logs/reinforcement_metrics.jsonl` for threshold crossings, emit summaries/alerts, and wire them into the diagnostics runner.
  3. [x] Upgrade repair helpers so missing configs can be regenerated from templates and schema mismatches are validated before returning `OK`.
  4. [x] Introduce a CI-friendly check (`scripts/ci_sanity.py` / `python -m scripts.continuous_probes --iterations 1 ...`) plus dry-run retraining to guard against regression prior to overnight cycles.
  5. [x] Tame self-preoccupation during canaries: tighten controller/self-bias dampers so repair runs can promote without manual intervention.
 6. [ ] Refresh probe baselines by retiring legacy low-authenticity logs once new heuristics land, keeping log sentinels focused on current behaviour. (Pinned requirements + LLM client retries make the overnight harness reproducible; leave warning until canaries are re-run.)
  7. [x] Teach the mid-span harness to emit per-run `summary_compact.json` files and live streak warnings so we can read auth/self/drift without chasing every JSONL entry.
  
- [x] **Diagnostics harness**: `scripts/diagnostics.py` now emits machine-readable results, covers hormones↔memory, router↔sampling, reinforcement feedback, and HTTP endpoint probes.
- [x] **Behaviour validation loop**: standardise five-cycle and multi-turn probes, trend authenticity, assistant_drift, self_preoccupation, and craving metrics before promotions; guardrail script now enforces session resets, idle cool-downs, and per-batch JSONL metrics with end-of-run summaries.

- [ ] **Mistral base pipeline experiment**: integrate `mistral-inference` with the `Mistral-7B-v0.3.Q4_K_M.gguf` base model, tune sampling knobs, and compare helper-drift metrics against the instruct pipeline.
- [x] **Safety and observability instrumentation**: live telemetry endpoints/console track hormones, controller deltas, and promotion thresholds in real time; overnight harness launches telemetry alongside FastAPI and rotates probe logs per batch for offline review.
- [x] **Diagnostic Repair Sequence upgrade**: diagnostics auto-attempt repairs, capture metadata per check, and support JSON output for CI gating.

- [x] **Endocrine feedback rebuild** *(11/4/2025)*:
  1. [x] **Instrumentation** – log hormone vectors, reinforcement channels, sampling params, and memory deltas per turn; extend probes (e.g. `scripts/bench_profiles`) to capture both instruct/base runs.
  2. [x] **Hormone dynamics model** – learn `h_{t+1} = f(h_t, reinforcement, intent, stimulus)` from logs; replace static adjustments with the learned forward pass in runtime.
  3. [x] **Reinforcement calibration** – expand `score_response` to multi-dimensional scores and map them into hormone deltas via the learned model.
  4. [x] **Controller policy** — recurrent sampler controller now ingests traits, hormone deltas, and memory tags to emit temperature/top_p/logit nudges; wired into `_prepare_chat_request` with telemetry for retraining.
  5. [x] **Memory coupling** — memory events now store endocrine/controller traces, selector scoring responds to hormone spikes, and diary reflections encode structured deltas for retraining.
  6. [x] **Continuous probes** — `scripts/continuous_probes.py` runs nightly multi-profile benches, logs endocrine metrics to `logs/probe_runs.jsonl`, and emits promotion gates.


## Pending Post-Restart Actions (2025-11-04) ## COMPLETED 
- [x] Check memory headroom after reboot prior to intensive probes.
- [x] Rerun `python -m scripts.continuous_probes` for two consecutive 5×5 suites to firm up authenticity/self-preoccupation trends.
- [x] Execute `python scripts/train_hormone_model.py` and `python scripts/train_controller_policy.py` against freshly captured logs.
- [x] Repeat the 5×5 probe loop and benchmark to validate metric movement before adjusting weights/parameters further.
- [x] Summarise deltas and decide on reinforcement weight tweaks targeting authenticity lift without exceeding self_preoccupation threshold.

## Completed Landmarks
- [x] **Mid-span probe harness (2025-11-06)**: `scripts/probes/mid_span_probes.py` delivers a configurable 30-60 minute injector with five scripted turns, cooldown controls, harness-scoped logging, and a compact JSON summary so quick reviews don’t require opening dozens of files.
### v0.17.4 Overnight Probe System
- [x] `scripts/probes/run_overnight.cmd` now runs the unattended probe cycle end-to-end: cleans stale locks, launches FastAPI with live telemetry, waits out warm-up, and streams guarded probe runs with configurable cadence/hard stops.
- [x] Canary probes, log sentinels, and repair helpers integrate with the overnight harness so authenticity/self-preoccupation thresholds raise alerts, stale logs rotate, and dry-run repairs can auto-attempt remediation before promotion.
- [x] Probe runners gained duration/idle guards, session resets between cycles, per-batch JSONL metrics, and end-of-run summaries, turning nightly benches into a repeatable regression gate for release candidates.
- [x] Mid-span harness telemetry now includes `run_meta.json` timing snapshots plus the standalone `mid_span_report.py` CLI for reviewing authenticity/self/outward streak sparklines, recent iterations, and gate hits immediately after each benchmark.

### v0.17.3 Emergent Tone Cleanup (archive/roadmaps/emergent-self-focus.md)
- [x] Persona mappings describe raw sensations (tight jaw, heavy limbs) instead of metaphorical tides.
- [x] Intent prompts and heuristics emphasise behavioural goals without imagery or stylistic guidance.
- [x] Reinforcement hunts collaborative phrasing ("we can", "your request") and logs actual trait/hormone snapshots.
- [x] Sampling previews default to on so tension-driven parameter shifts stay visible in logs.

### v0.17.2 Immersion Routing
- [x] Intent fragments steer the model toward internal sensations rather than validation scripts.
- [x] Heuristic fallbacks open with self-observation and drop listening or assistance assurances.
- [x] High tension widens temperature and top_p while trimming repetition penalties to encourage raw exploration.
- [x] Reinforcement metrics (`self_preoccupation`, hormone dampening) tighten feedback when self-focus sags; metrics plotting script added.

### v0.17.1 Groundwork
- [x] Reward structure favours self-referential authenticity and penalises assistant drift, feeding hormone feedback.
- [x] Internal reflections are tagged (`internal`, `reflection`) and prioritised in memory selection.
- [x] Trait-driven sampling swings and self-observation logit biases captured along with authenticity and sampling snapshots.

### v0.16 Streaming & Controls
- [x] Hand-tuned intent classifier with confidences and rationales replaced the heuristic router.
- [x] `/chat/stream` SSE endpoint and streaming client deliver incremental tokens with metadata.
- [x] Admin reload and caps endpoints plus hot config refresh; safeguards against `User:/Assistant:` artefacts.

### v0.15.x Local Engine Reliability
- [x] 0.15.1 ensured FastAPI boots the local llama engine, logging diagnostics when configuration is missing.
- [x] Detached launcher scripts, console handlers, and environment helpers stabilised the GPU-backed path.
- [x] Ambient stimuli (0.15) introduced time-based hormone noise and persona feedback loops.

### v0.10 - v0.14 Foundations
- [x] Noradrenaline hormone support and sampling parameter derivation from hormone deltas.
- [x] Temporal working-memory window with decay-based expiry checks and tests.
- [x] Reinforcement feedback loop mapping lexical metrics into hormone adjustments.
- [x] Output headroom adjustments (token caps, timeouts) and ASCII-normalised UI scaffolding.

### Earlier Sessions
- [x] GPU integration groundwork, llama.cpp build automation, and persona/UI refreshes captured in `docs/planning/archive/updates/0.10` - `0.15.1`.
- [x] Use these archives for historical context; active planning now lives in this cascade.
