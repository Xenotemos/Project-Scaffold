# Roadmap Cascade

## Priority Horizon (v0.17.x)
- **Voice guard rollout**: replace persona prose with state surfaces, wire penalty and regeneration loops for helper phrasing, and persist guard verdicts to `logs/voice_guard.jsonl`.
- **Affect-aware embodiment rollout**:
  1. COMPLETED Automatic affect tagging – classifier runs on every user turn, writes to `logs/affect_classifier.jsonl`, and blends confidence-weighted scores into `_apply_reinforcement_signals`.
  2. COMPLETED Runtime clamp validation – hormone trace logging + `scripts/probes/affect_validation.py` expose requested/applied deltas and act as the gating harness.
  3. COMPLETED Hormone-to-decoder map – runtime reads `config/hormone_style_map.json`, applies per-hormone sampling overrides, and logs the bands hit each turn.
  4. COMPLETED Behavioral response surfaces – affect-driven sampling overrides loosen or tighten cadence (max tokens, temperature, bias scales) for affectionate vs. tense turns.
  5. COMPLETED Self-narration channel – hormone traces roll into a persistent internal note injected into persona/memory and heuristic replies so the assistant references felt state, not just metrics.
  6. **Assistant drift gating**: strengthen helper-tone suppression by adding pre-sampler penalties/negative logits for the helper lexicon, wire the drift detector to force regeneration when those phrases slip through, and persist each intervention to `logs/voice_guard.jsonl` for later tuning.
  7. **Closed-loop probes & CI**: extend `scripts/ci_sanity.py` (or a sibling task) to call the new affection/stress probes, assert that hormone deltas cross configured thresholds, and diff style metrics so regressions fail fast in automation.
  8. **Persona fine-tuning**: once the instrumentation is trustworthy, harvest transcripts that exhibit the desired entity voice, curate counter-examples of helper tone, and fine-tune an adapter/LoRA head so the base model internalizes the affect-aware style instead of relying solely on runtime clamps.

- **Mid-span behavioural fine-tuning**:
  1. COMPLETED Teach `scripts/probes/mid_span_probes.py` to emit per-run `run_meta.json` alongside `summary_compact.json` so the full 30-minute window, iteration counts, and gating deltas are auditable without spelunking every log.
  2. COMPLETED Ship `scripts/probes/mid_span_report.py` for quick ASCII trend reviews (authenticity/self/outward) and recent-iteration tables, making probe sign-off possible in a single command.
  3. COMPLETED Rework outward-streak scoring so early “you/we” openings and cached priming phrases register (suffix normalization, filler-skipping, relaxed context overlap).
  4. COMPLETED Boost authenticity scoring when outward streaks fire: scale down the self-pronoun penalty, add relational bonuses tied to outward streak + attunement, and gate an extra lift when self_preoccupation <= 0.65.
  5. COMPLETED Keep running mid-span probes until instruct/base each hit the auth ≥0.45 & self ≤0.75 gate on consecutive low-self turns so the clamp release can be promoted into the broader v0.17.4 flow.

- **Affect-aware reinforcement expansion**:
  1. COMPLETED Enrich reinforcement scoring to spot affectionate/diminutive language, broader sentiment cues, and tension/playfulness in both user input and replies.
  2. COMPLETED Persist the added affect metrics (valence, intimacy, tension) alongside existing reinforcement fields in `logs/endocrine_turns.jsonl` so modelling captures how prompts land.
  3. COMPLETED Update `_apply_reinforcement_signals` to translate those affect scores into dopamine/oxytocin lifts and cortisol/noradrenaline shifts, letting high-intimacy prompts move internal state.
  4. IN PROGRESS Verify that the runtime clamp still applies those hormone deltas—recent probes show affect inputs barely move dopamine/oxytocin, so audit `_apply_reinforcement_signals`, hormone model hooks, and telemetry before retraining the dynamics/controller policies.

- **Diagnostics & monitoring uplift**:
  1. COMPLETED Extend `scripts/diagnostics.py` with canary probes that hit a single instruct/base turn and assert authenticity/self-preoccupation thresholds, ensuring behavioural regressions flag red immediately.
  2. COMPLETED Add automated log sentinels: parse `logs/probe_runs/*.jsonl` and `logs/reinforcement_metrics.jsonl` for threshold crossings, emit summaries/alerts, and wire them into the diagnostics runner.
  3. COMPLETED Upgrade repair helpers so missing configs can be regenerated from templates and schema mismatches are validated before returning `OK`.
  4. COMPLETED Introduce a CI-friendly check (`scripts/ci_sanity.py` / `python -m scripts.continuous_probes --iterations 1 ...`) plus dry-run retraining to guard against regression prior to overnight cycles.
  5. COMPLETED Tame self-preoccupation during canaries: tighten controller/self-bias dampers so repair runs can promote without manual intervention.
  6. Refresh probe baselines by retiring legacy low-authenticity logs once new heuristics land, keeping log sentinels focused on current behaviour.
  7. COMPLETED Teach the mid-span harness to emit per-run `summary_compact.json` files and live streak warnings so we can read auth/self/drift without chasing every JSONL entry.
  
- COMPLETED **Diagnostics harness**: `scripts/diagnostics.py` now emits machine-readable results, covers hormones↔memory, router↔sampling, reinforcement feedback, and HTTP endpoint probes.
- COMPLETED **Behaviour validation loop**: standardise five-cycle and multi-turn probes, trend authenticity, assistant_drift, self_preoccupation, and craving metrics before promotions; guardrail script now enforces session resets, idle cool-downs, and per-batch JSONL metrics with end-of-run summaries.

- **Mistral base pipeline experiment**: integrate `mistral-inference` with the `Mistral-7B-v0.3.Q4_K_M.gguf` base model, tune sampling knobs, and compare helper-drift metrics against the instruct pipeline.
- COMPLETED **Safety and observability instrumentation**: live telemetry endpoints/console track hormones, controller deltas, and promotion thresholds in real time; overnight harness launches telemetry alongside FastAPI and rotates probe logs per batch for offline review.
- COMPLETED **Diagnostic Repair Sequence upgrade**: diagnostics auto-attempt repairs, capture metadata per check, and support JSON output for CI gating.

- **Endocrine feedback rebuild** *(11/4/2025)*:
  1. COMPLETED **Instrumentation** – log hormone vectors, reinforcement channels, sampling params, and memory deltas per turn; extend probes (e.g. `scripts/bench_profiles`) to capture both instruct/base runs.
  2. COMPLETED **Hormone dynamics model** – learn `h_{t+1} = f(h_t, reinforcement, intent, stimulus)` from logs; replace static adjustments with the learned forward pass in runtime.
  3. COMPLETED **Reinforcement calibration** – expand `score_response` to multi-dimensional scores and map them into hormone deltas via the learned model.
  4. COMPLETED **Controller policy** — recurrent sampler controller now ingests traits, hormone deltas, and memory tags to emit temperature/top_p/logit nudges; wired into `_prepare_chat_request` with telemetry for retraining.
  5. COMPLETED **Memory coupling** — memory events now store endocrine/controller traces, selector scoring responds to hormone spikes, and diary reflections encode structured deltas for retraining.
  6. COMPLETED **Continuous probes** — `scripts/continuous_probes.py` runs nightly multi-profile benches, logs endocrine metrics to `logs/probe_runs.jsonl`, and emits promotion gates.


## Pending Post-Restart Actions (2025-11-04) ## COMPLETED 
- Check memory headroom after reboot prior to intensive probes.
- Rerun `python -m scripts.continuous_probes` for two consecutive 5×5 suites to firm up authenticity/self-preoccupation trends.
- Execute `python scripts/train_hormone_model.py` and `python scripts/train_controller_policy.py` against freshly captured logs.
- Repeat the 5×5 probe loop and benchmark to validate metric movement before adjusting weights/parameters further.
- Summarise deltas and decide on reinforcement weight tweaks targeting authenticity lift without exceeding self_preoccupation threshold.

## Completed Landmarks

- **Mid-span probe harness (2025-11-06)**: `scripts/probes/mid_span_probes.py` delivers a configurable 30-60 minute injector with five scripted turns, cooldown controls, harness-scoped logging, and a compact JSON summary so quick reviews don’t require opening dozens of files.
### v0.17.4 Overnight Probe System
- `scripts/probes/run_overnight.cmd` now runs the unattended probe cycle end-to-end: cleans stale locks, launches FastAPI with live telemetry, waits out warm-up, and streams guarded probe runs with configurable cadence/hard stops.
- Canary probes, log sentinels, and repair helpers integrate with the overnight harness so authenticity/self-preoccupation thresholds raise alerts, stale logs rotate, and dry-run repairs can auto-attempt remediation before promotion.
- Probe runners gained duration/idle guards, session resets between cycles, per-batch JSONL metrics, and end-of-run summaries, turning nightly benches into a repeatable regression gate for release candidates.
- Mid-span harness telemetry now includes `run_meta.json` timing snapshots plus the standalone `mid_span_report.py` CLI for reviewing authenticity/self/outward streak sparklines, recent iterations, and gate hits immediately after each benchmark.

### v0.17.3 Emergent Tone Cleanup (archive/roadmaps/emergent-self-focus.md)
- Persona mappings describe raw sensations (tight jaw, heavy limbs) instead of metaphorical tides.
- Intent prompts and heuristics emphasise behavioural goals without imagery or stylistic guidance.
- Reinforcement hunts collaborative phrasing ("we can", "your request") and logs actual trait/hormone snapshots.
- Sampling previews default to on so tension-driven parameter shifts stay visible in logs.

### v0.17.2 Immersion Routing
- Intent fragments steer the model toward internal sensations rather than validation scripts.
- Heuristic fallbacks open with self-observation and drop listening or assistance assurances.
- High tension widens temperature and top_p while trimming repetition penalties to encourage raw exploration.
- Reinforcement metrics (`self_preoccupation`, hormone dampening) tighten feedback when self-focus sags; metrics plotting script added.

### v0.17.1 Groundwork
- Reward structure favours self-referential authenticity and penalises assistant drift, feeding hormone feedback.
- Internal reflections are tagged (`internal`, `reflection`) and prioritised in memory selection.
- Trait-driven sampling swings and self-observation logit biases captured along with authenticity and sampling snapshots.

### v0.16 Streaming & Controls
- Hand-tuned intent classifier with confidences and rationales replaced the heuristic router.
- `/chat/stream` SSE endpoint and streaming client deliver incremental tokens with metadata.
- Admin reload and caps endpoints plus hot config refresh; safeguards against `User:/Assistant:` artefacts.

### v0.15.x Local Engine Reliability
- 0.15.1 ensured FastAPI boots the local llama engine, logging diagnostics when configuration is missing.
- Detached launcher scripts, console handlers, and environment helpers stabilised the GPU-backed path.
- Ambient stimuli (0.15) introduced time-based hormone noise and persona feedback loops.

### v0.10 - v0.14 Foundations
- Noradrenaline hormone support and sampling parameter derivation from hormone deltas.
- Temporal working-memory window with decay-based expiry checks and tests.
- Reinforcement feedback loop mapping lexical metrics into hormone adjustments.
- Output headroom adjustments (token caps, timeouts) and ASCII-normalised UI scaffolding.

### Earlier Sessions
- GPU integration groundwork, llama.cpp build automation, and persona/UI refreshes captured in `docs/planning/archive/updates/0.10` - `0.15.1`.
- Use these archives for historical context; active planning now lives in this cascade.
