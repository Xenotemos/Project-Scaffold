import json, pathlib, numpy as np
from unicodedata import normalize
src = pathlib.Path(r"fine_tune/merged/train_ready_guardrail_v6_plus_subtle.jsonl")
out = pathlib.Path(r"fine_tune/merged/train_ready_guardrail_v6_plus_subtle.cleaned.jsonl")
trans = {0x201C:'"',0x201D:'"',0x2018:"'",0x2019:"'",0x2014:'--',0x2013:'-'}
kept=0
clamped=0
non_ascii=0
with src.open(encoding='utf-8') as fin, out.open('w', encoding='utf-8', newline='') as fout:
    for line in fin:
        if not line.strip():
            continue
        obj=json.loads(line)
        # normalize text
        txt=str(obj.get('text',''))
        txt = normalize('NFKD', txt).translate(trans)
        if any(ord(c)>127 for c in txt):
            non_ascii+=1
        obj['text']=txt
        prev=obj.get('prev_turns') or []
        if not isinstance(prev,list):
            prev=[]
        obj['prev_turns']=[normalize('NFKD', str(t)).translate(trans) for t in prev]
        safety=float(obj.get('safety',0.0))
        intimacy=float(obj.get('intimacy',0.0))
        if safety < -0.2 and intimacy > 0.45:
            intimacy = max(0.05, 0.35 + 0.4*(safety+0.2))
            clamped +=1
        obj['intimacy']=round(float(intimacy),4)
        fout.write(json.dumps(obj, ensure_ascii=False)+"\n")
        kept+=1
print({'kept':kept,'clamped':clamped,'non_ascii_remaining':non_ascii,'out':str(out)})
