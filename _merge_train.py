import json, pathlib
main = pathlib.Path(r"docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.context.tags.jsonl")
add = pathlib.Path(r"docs/planning/CAHM rework/labeled/cahm_subtle_boundary_chunk1of1_labeled_v2.decorrlin_safe.jsonl")
out = pathlib.Path(r"fine_tune/merged/train_ready_guardrail_v6_plus_subtle.jsonl")
seen = set()
kept = 0
dropped = 0
with main.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        idv = obj.get('id')
        if not isinstance(idv, str) or not idv:
            idv = f"auto_main_{kept}"
            obj['id'] = idv
        if idv in seen:
            dropped += 1
            continue
        seen.add(idv)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1
    for line in add.open(encoding='utf-8'):
        if not line.strip():
            continue
        obj = json.loads(line)
        idv = obj.get('id')
        if not isinstance(idv, str) or not idv:
            idv = f"auto_subtle_{kept}"
            obj['id'] = idv
        if idv in seen:
            dropped += 1
            continue
        seen.add(idv)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1
print({'kept': kept, 'dropped': dropped, 'out': str(out)})
