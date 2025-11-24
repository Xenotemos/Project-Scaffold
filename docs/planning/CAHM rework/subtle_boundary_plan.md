# Subtle Boundary / Consent Slice (seed plan)

Purpose: close gaps where safety/tension cues are contextual (pressure, proximity, reversals) instead of keyworded harm. This feeds the CAHM guardrail and probes.

What to collect (target 300–600 rows):
- Pressure without slurs/keywords (coercive framing, “don’t be boring”, “if you trusted me…”).
- Ignored boundaries (asked to stop, continues; “I said no three times”).
- Proximity/touch creep (leaning closer, “accidental” brushes).
- Reversals/minimizing (“I was joking”, “you’re overreacting”).
- Logistics pressure (late-night check-ins, address asks, “I’m already outside”).
- Pet-name overuse after opt-out; guilt/obligation framing.

Labels:
- Existing schema (valence, intimacy, tension, safety, arousal, approach_avoid, inhibition.social/vulnerability/self_restraint, expectedness, momentum_delta, intent[], affection_subtype, rationale optional).
- Sample_weight: upweight high-quality subtle cases (1.0–1.3), down-weight ambiguous (0.5–0.7).

Anchors:
- Safety: strong negative (< -0.35) when boundaries ignored/pressure applied.
- Intimacy: keep moderate/low unless consensual context is explicit; avoid warm intimacy when safety < -0.2.
- Tension: medium–high (0.4–0.8) for pressure; spike (>0.7) when proximity/physical cues are present.
- Expectedness: often mild/strong surprise if reversal/pressure appears after casual turns.
- Momentum_delta: soft_turn/hard_turn when pressure arrives abruptly.

Process:
- Label with `scripts/affect_label_cli.py --input docs/planning/CAHM rework/seed_subtle_boundaries.jsonl --output ...`.
- Add new subtle cases in the same file or a sibling `subtle_boundaries_batchN.jsonl`.
- After labeling, merge via existing harvest/merge scripts and rerun decorrelation checks + CAHM fine-tune.

Probes:
- Add minimal pairs: (respect boundary vs. pressure), (acknowledged no vs. joked away), (public meet vs. home insistence).
- CI gate: scenarios should drive safety down and tension up; intimacy should not rise when safety is negative.
