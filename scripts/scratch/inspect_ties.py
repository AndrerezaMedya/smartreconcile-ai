import json, re

def normalize(t):
    t = t.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def jaccard(a, b):
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

with open('data/val.json', encoding='utf-8') as f:
    qs = json.load(f)

targets = ['VAL-VA-0018', 'VAL-VA-0021', 'VAL-VA-0024']
for q in qs:
    if q['query_id'] in targets:
        print(q['query_id'], q['difficulty'])
        print('  invoice:', q['invoice_line'])
        scores = []
        for c in q['candidates']:
            s = jaccard(q['invoice_line'], c['description'])
            scores.append((s, c['po_line_id'], c['is_correct'], c['description']))
        scores.sort(key=lambda x: -x[0])
        for s, pid, ic, desc in scores:
            tag = '[CORRECT]' if ic else '         '
            print(f'  {s:.4f} {tag} {desc[:55]}')
        print()
