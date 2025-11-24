import json
import scripts.affect_head_monitor as mon
line = '{"event": "affect_classification", "text_preview": "Hello, love", "source": "affect_head", "engine": "llama_cpp", "scores": {"valence": 0.4, "intimacy": 0.55, "tension": 0.05, "confidence": 0.85}, "tags": [], "latency_ms": 461.0, "raw_completion": null, "reasoning": null, "rationale": null, "ts": "2025-11-24T17:25:38Z", "extras": {"safety": 0.4, "arousal": 0.1, "approach_avoid": 0.3}}'
e=json.loads(line)
from inspect import signature
scores = e.get('scores') or {}
val = float(scores.get('valence', 0.0) or 0.0)
intimacy = float(scores.get('intimacy', 0.0) or 0.0)
tension = float(scores.get('tension', 0.0) or 0.0)
conf = float(scores.get('confidence', 0.0) or 0.0)
print(val, intimacy, tension, conf)
mon._render(e)
