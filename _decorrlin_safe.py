import json, pathlib, numpy as np, statistics
p = pathlib.Path(r"docs/planning/CAHM rework/labeled/cahm_subtle_boundary_chunk1of1_labeled_v2.decorrlin.jsonl")
rec = [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
vals = np.array([r['valence'] for r in rec])
ints = np.array([r['intimacy'] for r in rec])
saf = np.array([r['safety'] for r in rec])
# clamp high intimacy when unsafe to avoid unsafe+intimate cases
mask = (saf < -0.2) & (ints > 0.45)
ints[mask] = 0.35 + 0.3 * (saf[mask] + 0.2)  # push down toward ~0.29 when safety=-0.5
ints = np.clip(ints, 0.0, 1.0)
for r,val in zip(rec, ints):
    r['intimacy'] = float(round(val,4))
out = p.with_name('cahm_subtle_boundary_chunk1of1_labeled_v2.decorrlin_safe.jsonl')
with out.open('w', encoding='utf-8') as f:
    for r in rec:
        f.write(json.dumps(r, ensure_ascii=False)+'\n')
import math
cv=float(np.corrcoef(vals, ints)[0,1]); cvt=float(np.corrcoef(vals, np.array([r['tension'] for r in rec]))[0,1]);
mask2=(saf < -0.2) & (ints > 0.45)
print({'corr_val_int': round(cv,3), 'unsafe_high_int': int(mask2.sum()), 'out': str(out)})
