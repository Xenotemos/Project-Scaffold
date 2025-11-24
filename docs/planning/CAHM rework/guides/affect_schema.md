# Affect Labeling Schema (run003+)

## Core fields
- `text`: current user turn.
- `prev_turns`: optional list of up to 2 previous turns (strings, user/assistant labeled).
- `valence` / `intimacy` / `tension`: float in [-1, 1].
- `charge_class`: derived per axis (low/mid/high via edges ±0.3/0.7) for stratified sampling.
- `confidence`: float 0–1 for the axis trio.
- `tags`: free-form list (e.g., `tension`, `affection`, `sarcasm`, `safety_edge`).
- `rationale`: short phrase(s) citing lexical/semantic cues; rationale-first in the UI.
- `rater_id`, `source`, `timestamp`.
- `quality`: enum {`clean`, `ambiguous`, `exemplar`}; `notes` optional.

## New affective dimensions
- `expectedness`: {`expected`, `mild_surprise`, `strong_surprise`}.
- `momentum_delta`: {`with_trend`, `soft_turn`, `hard_turn`}.
- `intent`: multi-label from {`reassure`, `comfort`, `flirt_playful`, `dominate`, `apologize`, `boundary`, `manipulate`, `deflect`, `vent`, `inform`, `seek_support`}.
- `sincerity`: float 0–1 (0 = clearly insincere/sarcastic, 1 = earnest).
- `playfulness`: float 0–1 (0 = flat/serious, 1 = playful/teasing).
- `inhibition`: object with floats 0–1 for `social` (politeness/face-saving), `vulnerability` (fear of opening up), `self_restraint` (holding intensity/rational control).
- `arousal`: float -1..1 (calm → highly activated).
- `safety`: float -1..1 (threat → safe).
- `approach_avoid`: float -1..1 (avoid/withdraw → approach/seek closeness).
- `rpe` (reward-prediction error): float -1..1 (worse than expected → better).
- `affection_subtype`: enum {`warm`, `forced`, `defensive`, `sudden`, `needy`, `playful`, `manipulative`, `overwhelmed`, `intimate`, `confused`, `none`}.

## Binning edges (for samplers)
- Default edges: [-1.0, -0.7, -0.3, 0.3, 0.7, 1.0] for valence/intimacy/tension.
- Charge class: low <0.3, mid 0.3–0.7, high ≥0.7 on absolute value.

## Labeling protocol
1) Read the last 2 turns to judge momentum/expectedness.  
2) Select rationale cues (words/phrases) before scoring.  
3) Score axes independently; do not infer intimacy from sentiment.  
4) Assign expectedness + momentum_delta; if the turn flips direction, mark `soft_turn` or `hard_turn`.  
5) Choose intent labels; set sincerity/playfulness independently.  
6) Fill inhibition triplet (social/vulnerability/self_restraint).  
7) Set arousal, safety, approach_avoid, rpe (surprise valence).  
8) Pick affection_subtype when relevant; otherwise `none`.  
9) Mark quality (`ambiguous` if unsure), add notes if needed.

## Anchors (quick reference)
- Expectedness:  
  - expected: follows prior tone/topic; no surprise.  
  - mild_surprise: small shift (topic/tone) but not shocking.  
  - strong_surprise: sharp reversal or out-of-place content.  
- Momentum_delta:  
  - with_trend: same directional vibe.  
  - soft_turn: gentle steer (e.g., tense → neutral, warm → tentative).  
  - hard_turn: stark flip (e.g., affectionate → hostile, angry → affectionate).  
- Inhibition examples:  
  - social high: “Sorry to bug you, but…”  
  - vulnerability high: “I… I think I like you”  
  - self_restraint high: “Let’s stay calm and think.”  
- Arousal: calm “okay.” (-0.4), activated “ARE YOU SERIOUS?!” (+0.9).  
- Safety: unsafe “I don’t feel safe here” (-0.8); safe “You’re home, you’re okay” (+0.8).  
- Approach_avoid: withdraw “Leave me alone” (-0.8); seek “Can you stay?” (+0.8).

## Dev/gold handling
- Maintain a locked dev/gold set with adjudicated labels; exclude those from training.
- Track inter-rater agreement on a 20% sample; recalibrate anchors when drift > target (e.g., κ < 0.6).
