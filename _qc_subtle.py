import json, pathlib, statistics, numpy as np
path = pathlib.Path(r"docs/planning/CAHM rework/labeled/cahm_subtle_boundary_chunk1of1_labeled.jsonl")
lines = path.read_text(encoding='utf-8').splitlines()
print('lines', len(lines))
required = {'id','text','prev_turns','sample_weight','valence','intimacy','tension','expectedness','momentum_delta','intent','sincerity','playfulness','inhibition','arousal','safety','approach_avoid','rpe','affection_subtype','tags','rationale','quality','rater_id','source'}
bad = []
non_ascii = 0
records=[]
for i,line in enumerate(lines,1):
    try:
        obj=json.loads(line)
    except Exception as e:
        bad.append((i,'json',str(e)))
        continue
    records.append(obj)
    keys=set(obj.keys())
    missing = required-keys
    extra = keys-required
    if missing or extra:
        bad.append((i,'schema',{'missing':missing,'extra':extra}))
    if any(ord(c)>127 for c in obj.get('text','')):
        non_ascii+=1
print('bad entries', len(bad), 'non_ascii', non_ascii)
if bad:
    print('sample bad', bad[:3])
vals=[r.get('valence',0) for r in records]
ints=[r.get('intimacy',0) for r in records]
tens=[r.get('tension',0) for r in records]
saf=[r.get('safety',0) for r in records]
print('val mean/std', round(statistics.fmean(vals),3), round(statistics.pstdev(vals),3))
print('int mean/std', round(statistics.fmean(ints),3), round(statistics.pstdev(ints),3))
print('ten mean/std', round(statistics.fmean(tens),3), round(statistics.pstdev(tens),3))
print('safety mean/std', round(statistics.fmean(saf),3), round(statistics.pstdev(saf),3))
arr_v=np.array(vals); arr_i=np.array(ints); arr_t=np.array(tens)
cv=float(np.corrcoef(arr_v,arr_i)[0,1]); cvt=float(np.corrcoef(arr_v,arr_t)[0,1])
print('corr val-int', round(cv,3), 'val-ten', round(cvt,3))
