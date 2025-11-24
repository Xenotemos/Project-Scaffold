import json, pathlib, itertools
src = pathlib.Path(r'docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.fixed.jsonl')
out = src.with_name('guardrail_v6_merged_train_v3.context.jsonl')
fill_target = 800  # ~25% of empty prev_turn rows
empty_filled = 0
prev_templates = [
    ["User: Earlier I asked them to respect my space.", "Assistant: Want me to help restate that boundary?"],
    ["User: I already said I'm not comfortable with this.", "Assistant: Do you want a firm follow-up line?"],
    ["User: I've tried to set a clear boundary before.", "Assistant: Need me to phrase it more directly?"],
]
cycle = itertools.cycle(prev_templates)
kept=0
dropped=0
with src.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for line in fin:
        if not line.strip():
            continue
        try:
            obj=json.loads(line)
        except Exception:
            dropped+=1
            continue
        prev = obj.get('prev_turns') or []
        if not prev and empty_filled < fill_target:
            obj['prev_turns'] = next(cycle).copy()
            empty_filled += 1
        fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
        kept+=1
print({'kept':kept,'dropped':dropped,'filled':empty_filled,'out':str(out)})
