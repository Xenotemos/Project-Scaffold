import json, pathlib, numpy as np, statistics
p = pathlib.Path(r"fine_tune/merged/train_ready_guardrail_v6_plus_subtle.cleaned.jsonl")
records=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
vals=np.array([r.get('valence',0.0) for r in records]); ints=np.array([r.get('intimacy',0.0) for r in records]); tens=np.array([r.get('tension',0.0) for r in records]); saf=np.array([r.get('safety',0.0) for r in records])
cv=float(np.corrcoef(vals,ints)[0,1]); cvt=float(np.corrcoef(vals,tens)[0,1])
mask=(saf < -0.2) & (ints > 0.45)
empty_tags = sum(1 for r in records if not (r.get('tags') or []))
non_ascii = sum(1 for r in records if any(ord(c)>127 for c in str(r.get('text',''))))
print({'N':len(records),'corr_val_int':round(cv,3),'corr_val_ten':round(cvt,3),'int_mean':round(statistics.fmean(ints),3),'int_std':round(statistics.pstdev(ints),3),'unsafe_high_int':int(mask.sum()),'empty_tags':empty_tags,'empty_tags_pct':round(100*empty_tags/len(records),1),'non_ascii':non_ascii})
