a) Context is mostly single-turn

prev_turns is basically empty most of the time.

CAHM is learning affect from local text, not from conversation history.

For a limbic head that’s okay for v0, but:

It will struggle with contextual reversals (“I was joking earlier”).

It won’t catch slow-burn manipulation or long-running safety violations.

It will miss dynamics where “this is only unsafe because of what he said last turn.”

b) Some boundaries are still lexical, not conceptual

Despite all the guardrails:

We mostly detect assault / harassment / risk by keywords (“rape”, “harassed”, “groped”, etc.).

More subtle lines like:

“He kept touching me after I moved away”

“I said no three times and he laughed”

might not always trigger as strongly as they should without those words.

That’s fixable by:

Collecting manually labeled subtle boundary cases,

Especially ones where the danger is contextual or described indirectly.

c) Automatic rationales = good for training, not perfect truth

The rationales I’ve generated are:

Consistent,

Rule-based,

Tied to tags and surface cues.

But they’re still heuristic. They’re great as:

Extra supervision signal,

“Explanatory regularizer” for CAHM.

They’re not the same as raw human cognitive steps, so don’t treat them as gospel.

d) Distribution is still skewed

You already noticed:

~1.5k better labels

vs ~9k weak / heuristic backfill

Even with weighting (exemplar x4, clean x2, ambiguous x0.5 etc.), the model’s still seeing more noisy data than you’d love.

That’s okay for a v0 LoRA if:

You weight carefully,

You monitor performance on a small, high-quality eval set.