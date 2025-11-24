import pathlib
path = pathlib.Path(r'docs/planning/CAHM rework/labeled/merged_guardrail_v6/guardrail_v6_merged_train_v3.jsonl')
bad = ("â", "Â", "�", "\\u2019", "\\u00e2")
issues = []
with path.open(encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if any(b in line for b in bad):
            issues.append((i, line.strip()[:180]))
print('found', len(issues))
for i, txt in issues[:10]:
    print(f"{i}: {txt}")
