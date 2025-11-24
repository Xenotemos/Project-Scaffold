import json, pathlib
src = pathlib.Path(r'docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.jsonl')
out = src.with_name('guardrail_v6_merged_train_v3.cleaned.jsonl')
bad_chars = ('â','Â','�')
kept = 0; dropped = 0; null_id = 0; bad_text = 0; bad_prev = 0
with src.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for line in fin:
        line=line.strip()
        if not line:
            continue
        try:
            obj=json.loads(line)
        except Exception:
            dropped+=1; continue
        idv = obj.get('id')
        if not isinstance(idv,str) or not idv.strip():
            null_id+=1; dropped+=1; continue
        text=str(obj.get('text',''))
        if any(b in text for b in bad_chars):
            bad_text+=1; dropped+=1; continue
        prev=obj.get('prev_turns') or []
        if not isinstance(prev,list) or not all(isinstance(t,str) for t in prev):
            bad_prev+=1; dropped+=1; continue
        fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
        kept+=1
print({'kept':kept,'dropped':dropped,'null_id':null_id,'bad_text':bad_text,'bad_prev':bad_prev,'out':str(out)})
