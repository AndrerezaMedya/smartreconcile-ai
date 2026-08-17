import json

with open("data/material_catalog.json", encoding="utf-8") as f:
    cat = json.load(f)

for m in cat["materials"]:
    if m["split"] == "test":
        print(f"[{m['material_id']}] {m['category']} | {m['canonical_name']} / {m['canonical_name_en']}")
        print(f"   Specs: {m.get('specifications', {})}")
        print(f"   Abbrs: {m.get('common_abbreviations', [])}")
        print(f"   Related: {m.get('related_but_different', [])}")
        print()
