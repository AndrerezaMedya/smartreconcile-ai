import json, re
from pathlib import Path

DATA_DIR = Path("data")

all_test = {}
for fname in ["test_synthetic_v2.json", "test_hard_neg.json", "test_adversarial_v2.json"]:
    p = DATA_DIR / fname
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for q in json.load(f):
                for c in q["candidates"]:
                    all_test[c["description"].strip().upper()] = fname
                all_test[q["invoice_line"].strip().upper()] = fname

def norm(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

with open("scripts/b1_improved_training_data.py", encoding="utf-8") as f:
    src = f.read()

# Extract all quoted uppercase strings from the script
strings_in_script = re.findall(r'"([A-Z][A-Z0-9 /.*\-x#]{4,})"', src)

print("Strings in B1 script matching test set:")
found = set()
for s in strings_in_script:
    if s.upper() in all_test:
        if s not in found:
            print(f"  EXACT: {s!r} in {all_test[s.upper()]}")
            found.add(s)
    else:
        ta = set(norm(s).split())
        for td, tsrc in all_test.items():
            tb = set(norm(td).split())
            if ta and tb and len(ta | tb) > 0:
                j = len(ta & tb) / len(ta | tb)
                if j >= 0.85:
                    key = (s, td)
                    if key not in found:
                        print(f"  NEAR j={j:.2f}: {s!r} ~= {td!r} [{tsrc}]")
                        found.add(key)
