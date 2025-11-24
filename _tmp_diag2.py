import json, time, os
from brain.affect_classifier import LlamaCppAffectClassifier
cfg=json.load(open("config/affect_classifier.json"))
cfg["timeout"]=10; cfg["soft_timeout"]=5; cfg["readiness_timeout"]=3
clf=LlamaCppAffectClassifier(cfg)
text="Hello, baby"
try:
    start=time.time()
    clf._ensure_ready()
    completion=clf._query_model(text)
    parsed=clf._parse_completion(completion)
    latency=time.time()-start
    print('completion', completion[:120])
    print('parsed', parsed)
    print('latency', latency)
except Exception as e:
    print('exception before construct', e)
    raise
