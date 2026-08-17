import json

with open('data/test_synthetic.json', encoding='utf-8') as f:
    qs = json.load(f)

targets = ['SYN-TE-0013','SYN-TE-0015','SYN-TE-0022','SYN-TE-0024','SYN-TE-0025']
for q in qs:
    if q['query_id'] not in targets:
        continue
    print('='*60)
    print(q['query_id'], '|', q['difficulty'], '|', q['category'])
    print('  invoice :', q['invoice_line'])
    print('  mat_id  :', q['material_id'])
    print('  vartypes:', q.get('variation_types'))
    for c in q['candidates']:
        tag = '[CORRECT]' if c['is_correct'] else '         '
        neg = c.get('neg_type', '') or ''
        print(f'  {tag} | {neg:22} | {c["description"]}')
    print()
