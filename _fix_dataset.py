import json, pathlib
src = pathlib.Path(r'docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.jsonl')
out = src.with_name('guardrail_v6_merged_train_v3.fixed.jsonl')
skip_range = range(1156, 1206)
bad_chars = ('â','Â','�')
kept = 0; dropped = 0
with src.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for idx, line in enumerate(fin, 1):
        if idx in skip_range:
            dropped += 1
            continue
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except Exception:
            dropped += 1
            continue
        text=str(obj.get('text',''))
        if any(b in text for b in bad_chars):
            dropped += 1
            continue
        if not isinstance(obj.get('id'), str) or not obj.get('id'):
            obj['id'] = f"auto_{idx:05d}"
        prev=obj.get('prev_turns') or []
        if not isinstance(prev, list):
            prev=[]
        obj['prev_turns'] = [str(t) for t in prev]
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1
print({'kept': kept, 'dropped': dropped, 'out': str(out)})
