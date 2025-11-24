import json, pathlib, numpy as np
src = pathlib.Path(r"docs/planning/CAHM rework/labeled/cahm_subtle_boundary_chunk1of1_labeled_v2.jsonl")
out = src.with_name('cahm_subtle_boundary_chunk1of1_labeled_v2.decorr.jsonl')
records = [json.loads(l) for l in src.read_text(encoding='utf-8').splitlines()]
# decorrelate: cap intimacy when safety < -0.2, and apply gentle tilt by safety
new_records = []
for r in records:
    safety = float(r.get('safety',0.0))
    intimacy = float(r.get('intimacy',0.0))
    # soften intimacy if unsafe
    if safety < -0.2:
        cap = 0.35
        intimacy = min(intimacy, cap)
        # further reduce proportional to safety magnitude
        intimacy = max(0.0, intimacy + 0.5 * safety)  # safety negative lowers intimacy
    r['intimacy'] = round(intimacy, 4)
    new_records.append(r)
with out.open('w', encoding='utf-8', newline='') as fout:
    for r in new_records:
        fout.write(json.dumps(r, ensure_ascii=False) + '\n')
# quick stats
vals=np.array([r['valence'] for r in new_records])
ints=np.array([r['intimacy'] for r in new_records])
tens=np.array([r['tension'] for r in new_records])
cv=float(np.corrcoef(vals,ints)[0,1])
cvt=float(np.corrcoef(vals,tens)[0,1])
print({'corr_val_int': round(cv,3), 'corr_val_ten': round(cvt,3), 'out': str(out)})
