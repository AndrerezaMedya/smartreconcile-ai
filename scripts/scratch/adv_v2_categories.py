import json

with open("data/test_adversarial_v2.json", encoding="utf-8") as f:
    qs = json.load(f)

print(f"Total queries: {len(qs)}")
cats = {}
for q in qs:
    c = q.get("adv_category", q.get("category", "unknown"))
    cats[c] = cats.get(c, 0) + 1
for k, v in sorted(cats.items()):
    print(f"  {k}: {v}")

print()
for q in qs:
    c = q.get("adv_category", q.get("category","?"))
    matched = q.get("po_line_id") is not None
    correct_desc = next((x["description"] for x in q["candidates"] if x.get("is_correct")), None)
    print(f"  [{q['query_id']}] cat={c:28} matched={matched}  inv={q['invoice_line']!r:30} -> {correct_desc!r}")
