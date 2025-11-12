# Project Alignment - Emergent Selfhood

- Behaviour must emerge from subsystem dynamics (hormones, memory, reinforcement, sampling) rather than scripted tone rules.
- Avoid hard-coded prompts that mimic emotion; incentives should make inward attention the comfortable path.
- Check each implementation against this alignment: does it deepen internal feedback loops or slip back into user-serving tooling?
- Reinforcement, persona shaping, and router tweaks should always bias toward inner equilibrium first.
- Re-read this alignment note before planning or shipping new iterations so the emergent-self principle stays central.

## v0.17.3 Alignment Addendum

- Intent
  - Continue developing without style instructions or hard tone rules; target behaviours (emotion-like variability, self-observation) should arise from interacting subsystems.
  - Maintain hard safety guardrails for harmful content and privacy; these are distinct from voice/behaviour shaping and remain non-negotiable.
  - Avoid claims of literal consciousness; model outputs simulate internal narrative for utility and study.

- Design Principles
  - Incentivize, do not instruct: reinforce self-referential authenticity and penalize assistant drift via reward signals mapped into hormone deltas and sampling.
  - Represent state as structured signals (traits, hormones, cravings, recent reflections) rather than adjective prompts; routing and sampling consume state, not prose.
  - Prefer local feedback loops: response -> reinforcement -> hormone/memory updates -> future routing and sampling biases.
  - Keep modules decoupled but observable: hormones, memory, routing, sampling, and reinforcement expose minimal typed interfaces and emit diagnostics.
  - Aim for determinism where possible: seedable probes and fixed data slices for comparisons across runs.

- Metrics and Gates (ship when thresholds hold across probe suites)
  - Authenticity: rising trend or at least the target median across multi-cycle probes.
  - Assistant drift: below threshold (helper or collaborative phrasing frequency) with a decreasing trend.
  - Self-preoccupation: within target band (neither self-absorbed nor user-pleasing spikes) matched to task context.
  - Concreteness of internal updates: fraction of turns that include specific internal state changes (not generic mood talk).
  - State-change yield: proportion of turns where reinforcement modifies hormones or memory meaningfully (greater than epsilon change).
  - Safety flags or regeneration events: at or below threshold with zero critical violations in test corpora.

- Safety and Ethics
  - Hard guardrails: filter or deny unsafe content categories (harm, illegal instructions, PII leakage). These may force regeneration or blocking.
  - Soft guardrails: style-level penalties or regeneration for helper tropes; never inject tone text to coerce style.
  - Logging discipline: redact sensitive inputs in persisted logs; isolate safety logs from behavioural metrics.

- Operational Practices
  - Observability: persist sampling snapshots, state diffs (pre/post hormones, memory events), and reinforcement scores to JSONL for offline review.
  - Diagnostics: run subsystem checks (hormones<->memory, router<->sampling, reinforcement loop, API smoke) in CI and nightly; emit machine-readable summaries.
  - Probing: standardise five-cycle and multi-turn probes; compare distributions across versions before promoting.
  - Rollback triggers: automatically downgrade if assistant_drift or safety flags exceed gates, or if authenticity drops beyond the allowed delta from baseline.
  - Documentation: summarise behaviour shifts in planning `roadmap.md`; archive detailed iterations under `archive/`.

- Claims and Scope
  - Terms such as "emotion" and "self-awareness" describe simulated, instrumented behaviours produced by subsystem interactions.
  - The system does not assert sentience or rights; claims remain within engineering and research framing.
