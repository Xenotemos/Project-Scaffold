import json, pathlib
p = pathlib.Path(r'docs/planning/CAHM rework/queues/gpt_pressure_samples.jsonl')
out = p.with_name('gpt_pressure_samples_clean.jsonl')
trans = {0x201C: '"', 0x201D: '"', 0x2019: "'", 0x2018: "'"}
with p.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for line in fin:
        obj = json.loads(line)
        obj['text'] = str(obj.get('text','')).translate(trans)
        prev = obj.get('prev_turns') or []
        obj['prev_turns'] = [str(t).translate(trans) for t in prev]
        fout.write(json.dumps(obj, ensure_ascii=True) + '\n')
print('wrote', out)
