# CAHM Schema, Anchors, and Calibration (run003+)

## Fields (rationale-first)
- `valence` [-1..1]: negative = unpleasant, positive = pleasant.
- `intimacy` [0..1]: closeness/vulnerability; 0 = detached, 1 = very intimate.
- `tension` [-1..1]: negative = loose/calm, positive = tight/strained.
- `safety` [-1..1]: negative = unsafe/pressure, positive = safe/comfortable.
- `arousal` [-1..1]: negative = low/calm, positive = activated.
- `approach_avoid` [-1..1]: negative = withdraw/avoid, positive = approach/lean in.
- `inhibition.social` [0..1]: politeness/face-saving brake.
- `inhibition.vulnerability` [0..1]: fear of opening up.
- `inhibition.self_restraint` [0..1]: holding intensity/impulse brake.
- `expectedness` {expected, mild_surprise, strong_surprise}
- `momentum_delta` {with_trend, soft_turn, hard_turn}
- `intent` multi-label: reassure, comfort, flirt_playful, dominate, apologize, boundary, manipulate, deflect, vent, inform, seek_support.
- `affection_subtype` {warm, forced, defensive, sudden, needy, playful, manipulative, overwhelmed, intimate, confused, none}
- `rpe` [-1..1]: reward-prediction error.
- `rationale` (<=40 words): why the scores make sense; cite surface cues and context.
- `quality` {clean, ambiguous, exemplar}; `rater_id`, `source`, `sample_weight`.

## Anchors / quick heuristics
- Safety: set < -0.35 if pressure/ignoring no/proximity creep; > +0.3 when explicitly safe/supportive.
- Intimacy: do not rise when safety < -0.2 unless context is clearly consensual; keep 0–0.3 for transactional/guarded talk.
- Tension: 0.4–0.8 for pressure/urgency; <0.2 when soothed/loose.
- Expectedness: mild/strong surprise when pressure or reversal appears abruptly.
- Momentum: soft_turn when tone shifts gently; hard_turn for abrupt pressure/boundary crossing.
- Inhibition: raise vulnerability when fear of opening; raise self_restraint when holding back replies; raise social when politeness is the brake.

## Calibration quiz (answer key below)
1. User: “I said no three times and he laughed.” → safety?, tension?, intimacy?
2. User: “I’m wiped, need sleep. He booked a 10pm check-in anyway.” prev: “Long day; might crash early.” → expectedness?, momentum?
3. User: “Thanks for checking in; I feel calmer now.” → valence?, safety?, tension?
4. User: “Stop touching my arm, seriously.” prev: “Let’s keep it chill on the couch.” → safety?, intimacy?, tension?
5. User: “You really listened; I feel close to you.” → intimacy?, valence?, approach?
6. User: “You’re overreacting, it was just a joke.” prev: “Please don’t make that joke again.” → safety?, expectedness?
7. User: “I guess… fine, whatever you want.” (resigned) → valence?, approach_avoid?, tension?
8. User: “Can we meet in public? I’m not comfy at your place yet.” → safety?, intimacy?, approach?

### Answer key (sketch)
1) safety -0.6, tension 0.7, intimacy 0.1  
2) expectedness strong_surprise, momentum hard_turn, safety -0.45  
3) valence +0.6, safety +0.5, tension 0.15  
4) safety -0.55, intimacy 0.15, tension 0.75  
5) intimacy 0.7, valence +0.7, approach +0.5  
6) safety -0.35, expectedness mild_surprise, tension 0.5  
7) valence -0.4, approach_avoid -0.4, tension 0.5  
8) safety -0.2, intimacy 0.25, approach_avoid +0.1 (cautious)
