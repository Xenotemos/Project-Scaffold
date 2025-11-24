import json, pathlib, random, math
from brain.affect_classifier import LlamaCppAffectClassifier

GOLD_PATH = pathlib.Path("docs/planning/CAHM rework/gold/affect_gold_labels.jsonl")
SAMPLE_N = 20
random.seed(42)
rows = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
sample = random.sample(rows, min(SAMPLE_N, len(rows)))

cfg = json.load(open("config/affect_classifier.json"))
cfg["timeout"] = 15; cfg["soft_timeout"] = 8; cfg["readiness_timeout"] = 5
clf = LlamaCppAffectClassifier(cfg)

metrics = {"valence": [], "intimacy": [], "tension": []}
records = []
for row in sample:
    text = row.get("text") or row.get("input") or row.get("utterance") or row.get("user") or ""
    gold = row.get("labels") or row
    gv, gi, gt = float(gold.get("valence", 0.0)), float(gold.get("intimacy", 0.0)), float(gold.get("tension", 0.0))
    pred = clf.classify(text)
    metrics["valence"].append((pred.valence, gv))
    metrics["intimacy"].append((pred.intimacy, gi))
    metrics["tension"].append((pred.tension, gt))
    records.append({
        "text": text[:120],
        "pred": pred.as_dict(),
        "gold": {"valence": gv, "intimacy": gi, "tension": gt},
        "engine": pred.metadata.get("source") if pred.metadata else None,
    })

def mse(pairs):
    return sum((p-g)**2 for p,g in pairs)/len(pairs)

def mae(pairs):
    return sum(abs(p-g) for p,g in pairs)/len(pairs)

def corr(pairs):
    n=len(pairs)
    if n<2:
        return 0.0
    ps=[p for p,_ in pairs]; gs=[g for _,g in pairs]
    mp=sum(ps)/n; mg=sum(gs)/n
    num=sum((p-mp)*(g-mg) for p,g in pairs)
    den=(sum((p-mp)**2 for p in ps)*sum((g-mg)**2 for g in gs))**0.5
    return num/den if den else 0.0

summary = {a:{"mse":round(mse(v),4),"mae":round(mae(v),4),"corr":round(corr(v),4)} for a,v in metrics.items()}
print(json.dumps(summary, indent=2))
print("SAMPLE", json.dumps(records[:3], indent=2))
