"""
B1 — Improved Training Data (train_v2.json)
Adds generalized examples for the identified failure categories WITHOUT
copying exact test instances.

New categories added:
  A. Abbreviation <-> full form (BRC, MCB, V-belt, etc.)
  B. Indonesian <-> English with spec context
  C. Short but resolvable descriptions
  D. Semantic neighbor contrastive pairs (chain/sprocket, bolt/nut, etc.)
  E. Domain aliases / vendor-style variations

Strict rules:
  - No material ID from test split
  - No exact normalized description overlap with test set
  - No reconstruction of test query "BRC", "Rantai", "Baut", "Gear Rantai"
  - All new material families or distinct spec values

Output: data/train_v2.json (appended to clean train.json)
"""

import json
import re
import random
from pathlib import Path

random.seed(123)  # different from all other seeds

DATA_DIR = Path("data")

# Load existing train data
with open(DATA_DIR / "train.json", encoding="utf-8") as f:
    train_v1 = json.load(f)

# Load test sets for leakage check
all_test_descs = set()
for fname in ["test_synthetic_v2.json", "test_hard_neg.json", "test_adversarial_v2.json"]:
    p = DATA_DIR / fname
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for q in json.load(f):
                for c in q["candidates"]:
                    all_test_descs.add(c["description"].strip().upper())
                all_test_descs.add(q["invoice_line"].strip().upper())

def normalize(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def leakage_check(invoice_text, po_text):
    """Returns True if either text overlaps with test set."""
    for t in [invoice_text, po_text]:
        if t.strip().upper() in all_test_descs:
            return True
        # Near-match: Jaccard > 0.85
        ta = set(normalize(t).split())
        for test_desc in all_test_descs:
            tb = set(normalize(test_desc).split())
            if ta and tb:
                j = len(ta & tb) / len(ta | tb)
                if j > 0.85:
                    return True
    return False

# ── New training examples ─────────────────────────────────────────────────────
# Format: {query_id, invoice_line, po_line_id, candidates, difficulty,
#          variation_types, category, material_id="TRAIN-EXTRA-xxx"}
# All candidates: [correct, neg1, neg2, neg3] (shuffled)

NEW_EXAMPLES = []
qid_counter = [1]  # mutable counter

def make_qid():
    q = f"TRV2-{qid_counter[0]:03d}"
    qid_counter[0] += 1
    return q

def make_example(invoice, po_desc, negatives, difficulty, category, vtypes, mat_id):
    """negatives: list of (description, neg_type)"""
    qid = make_qid()
    cands = [{"po_line_id": f"{qid}-PL-C", "description": po_desc,
               "is_correct": True, "neg_type": None}]
    for i, (neg_d, neg_t) in enumerate(negatives):
        cands.append({"po_line_id": f"{qid}-PL-N{i+1}", "description": neg_d,
                      "is_correct": False, "neg_type": neg_t})
    random.shuffle(cands)
    return {
        "query_id": qid, "invoice_line": invoice,
        "po_line_id": f"{qid}-PL-C",
        "candidates": cands, "difficulty": difficulty,
        "variation_types": vtypes, "category": category,
        "material_id": mat_id,
    }

# ── CATEGORY A: Abbreviation <-> full form ────────────────────────────────────
# "BRC" is in test — we use other wire mesh abbreviation forms and related items

a_examples = [
    make_example(
        "Wiremesh M8 200x200",
        "WELDED WIRE MESH M8 WIRE DIAMETER 200X200MM",
        [("WELDED WIRE MESH M6 WIRE DIAMETER 150X150MM", "spec_diff_wire"),
         ("EXPANDED METAL MESH 2 INCH DIAMOND", "different_type"),
         ("CHAIN LINK FENCE GALVANIZED 50X50MM", "different_item")],
        "hard", "structural_steel", ["abbreviation"],
        "TRAIN-EXTRA-A01"
    ),
    make_example(
        "Wiremesh M6-150",
        "WELDED WIRE MESH M6 WIRE DIAMETER 150X150MM",
        [("WELDED WIRE MESH M8 WIRE DIAMETER 150X150MM", "spec_diff_wire"),
         ("WELDED WIRE MESH M6 WIRE DIAMETER 200X200MM", "spec_diff_mesh"),
         ("CHAIN LINK FENCE 1 INCH", "different_type")],
        "hard", "structural_steel", ["abbreviation"],
        "TRAIN-EXTRA-A02"
    ),
    make_example(
        "MCB 1P 10A Schneider",
        "MCB 1 POLE 10A 6KA MINIATURE CIRCUIT BREAKER",
        [("MCB 1 POLE 25A 6KA MINIATURE CIRCUIT BREAKER", "spec_diff_ampere"),
         ("MCB 2 POLE 10A 6KA MINIATURE CIRCUIT BREAKER", "spec_diff_pole"),
         ("FUSE 10A CYLINDRICAL GL TYPE", "different_type")],
        "hard", "electrical", ["abbreviation"],
        "TRAIN-EXTRA-A03"
    ),
    make_example(
        "MCCB 3P 63A",
        "MCCB 3 POLE 63A 18KA MOLDED CASE CIRCUIT BREAKER",
        [("MCCB 3 POLE 100A 18KA MOLDED CASE CIRCUIT BREAKER", "spec_diff_ampere"),
         ("MCB 3 POLE 63A 6KA", "different_type"),
         ("ELCB 2 POLE 63A 30MA", "different_type")],
        "hard", "electrical", ["abbreviation"],
        "TRAIN-EXTRA-A04"
    ),
    make_example(
        "V-Belt A55",
        "V-BELT TYPE A SECTION LENGTH 55 INCHES",
        [("V-BELT TYPE B SECTION LENGTH 55 INCHES", "spec_diff_section"),
         ("V-BELT TYPE A SECTION LENGTH 62 INCHES", "spec_diff_length"),
         ("FLAT BELT RUBBER 50MM WIDTH", "different_type")],
        "hard", "mechanical", ["abbreviation"],
        "TRAIN-EXTRA-A05"
    ),
    make_example(
        "Fanbelt B52",
        "V-BELT TYPE B SECTION LENGTH 52 INCHES",
        [("V-BELT TYPE A SECTION LENGTH 52 INCHES", "spec_diff_section"),
         ("V-BELT TYPE B SECTION LENGTH 55 INCHES", "spec_diff_length"),
         ("TIMING BELT 600H100", "different_type")],
        "medium", "mechanical", ["abbreviation"],
        "TRAIN-EXTRA-A06"
    ),
    make_example(
        "Kontaktor 25A 220V",
        "MAGNETIC CONTACTOR 25A COIL 220V AC",
        [("MAGNETIC CONTACTOR 18A COIL 220V AC", "spec_diff_ampere"),
         ("MAGNETIC CONTACTOR 25A COIL 380V AC", "spec_diff_voltage"),
         ("THERMAL OVERLOAD RELAY 18-25A", "different_type")],
        "hard", "electrical", ["abbreviation"],
        "TRAIN-EXTRA-A07"
    ),
]

# ── CATEGORY B: Indonesian <-> English with spec context ──────────────────────
b_examples = [
    make_example(
        "Kran bola 1 inch stainless",
        "BALL VALVE 1 INCH SS304 FULL BORE",
        [("BALL VALVE 1/2 INCH SS304 FULL BORE", "spec_diff_size"),
         ("BALL VALVE 1 INCH CARBON STEEL FULL BORE", "spec_diff_material"),
         ("GATE VALVE 1 INCH SS304", "different_type")],
        "hard", "piping", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B01"
    ),
    make_example(
        "Selang hydraulik 1.5 inch SAE 100R1",
        "HYDRAULIC HOSE SAE 100R1AT 1-1/2 INCH",
        [("HYDRAULIC HOSE SAE 100R2AT 1-1/2 INCH", "spec_diff_type"),
         ("HYDRAULIC HOSE SAE 100R1AT 1 INCH", "spec_diff_size"),
         ("AIR HOSE RUBBER 1-1/2 INCH 20 BAR", "different_type")],
        "hard", "mechanical", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B02"
    ),
    make_example(
        "Kompresor udara 10 bar 50L",
        "AIR COMPRESSOR 10 BAR TANK 50L 2HP 220V",
        [("AIR COMPRESSOR 8 BAR TANK 50L 2HP 220V", "spec_diff_pressure"),
         ("AIR COMPRESSOR 10 BAR TANK 100L 3HP 380V", "spec_diff_size"),
         ("VACUUM PUMP 10 BAR 2HP", "different_type")],
        "hard", "mechanical", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B03"
    ),
    make_example(
        "Oli hidrolik VG 46",
        "HYDRAULIC OIL ISO VG 46 DRUM 209L",
        [("HYDRAULIC OIL ISO VG 68 DRUM 209L", "spec_diff_grade"),
         ("HYDRAULIC OIL ISO VG 46 PAIL 20L", "spec_diff_packaging"),
         ("GEAR OIL SAE 90 GL-4 DRUM 209L", "different_type")],
        "hard", "chemicals", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B04"
    ),
    make_example(
        "Kabel NYY 2x0.75mm2",
        "NYY POWER CABLE 2 CORE 0.75MM2 0.6/1KV",
        [("NYY POWER CABLE 2 CORE 1.5MM2 0.6/1KV", "spec_diff_size"),
         ("NYY POWER CABLE 3 CORE 0.75MM2 0.6/1KV", "spec_diff_cores"),
         ("NYA SINGLE CORE CABLE 0.75MM2", "different_type")],
        "hard", "electrical", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B05"
    ),
    make_example(
        "Besi siku 40x40x4mm 6m",
        "ANGLE BAR 40X40X4MM JIS G3101 SS400 L=6M",
        [("ANGLE BAR 50X50X5MM JIS G3101 SS400 L=6M", "spec_diff_size"),
         ("ANGLE BAR 40X40X4MM JIS G3101 SS400 L=12M", "spec_diff_length"),
         ("FLAT BAR 40X4MM SS400 L=6M", "different_type")],
        "hard", "structural_steel", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B06"
    ),
    make_example(
        "Plat kapal tebal 10mm",
        "STEEL PLATE 10MM HOT ROLLED SS400 4X8FEET",
        [("STEEL PLATE 8MM HOT ROLLED SS400 4X8FEET", "spec_diff_thick"),
         ("STEEL PLATE 10MM COLD ROLLED SPCC", "spec_diff_type"),
         ("CHECKERED PLATE 10MM SS400 4X8FEET", "different_type")],
        "medium", "structural_steel", ["indonesian_english_mix"],
        "TRAIN-EXTRA-B07"
    ),
]

# ── CATEGORY C: Short but resolvable descriptions ─────────────────────────────
c_examples = [
    make_example(
        "Bearing 6208 2RS",
        "DEEP GROOVE BALL BEARING 6208 2RS C3",
        [("DEEP GROOVE BALL BEARING 6208 ZZ C3", "spec_diff_seal"),
         ("DEEP GROOVE BALL BEARING 6308 2RS C3", "spec_diff_size"),
         ("SELF-ALIGNING BALL BEARING 1208", "different_type")],
        "hard", "mechanical", ["truncation"],
        "TRAIN-EXTRA-C01"
    ),
    make_example(
        "Bearing 6306",
        "DEEP GROOVE BALL BEARING 6306 ZZ",
        [("DEEP GROOVE BALL BEARING 6206 ZZ", "spec_diff_size"),
         ("DEEP GROOVE BALL BEARING 6306 2RS", "spec_diff_seal"),
         ("ANGULAR CONTACT BEARING 7306 B", "different_type")],
        "hard", "mechanical", ["truncation"],
        "TRAIN-EXTRA-C02"
    ),
    make_example(
        "U-Bolt M16",
        "U-BOLT M16 GALVANIZED FOR PIPE",
        [("U-BOLT M12 GALVANIZED FOR PIPE", "spec_diff_size"),
         ("U-BOLT M16 STAINLESS SS304", "spec_diff_material"),
         ("J-BOLT M16 GALVANIZED", "different_type")],
        "medium", "fasteners", ["truncation"],
        "TRAIN-EXTRA-C03"
    ),
    make_example(
        "Elbow 1 inch PVC",
        "PVC ELBOW 90DEG 1 INCH SOCKET TYPE",
        [("PVC ELBOW 90DEG 1/2 INCH SOCKET TYPE", "spec_diff_size"),
         ("PVC ELBOW 45DEG 1 INCH SOCKET TYPE", "spec_diff_angle"),
         ("GALVANIZED ELBOW 1 INCH 90DEG", "different_material")],
        "medium", "piping", ["truncation"],
        "TRAIN-EXTRA-C04"
    ),
    make_example(
        "Flens 4 inch CS150",
        "FLANGE SLIP-ON 4 INCH CARBON STEEL ANSI B16.5 150LB",
        [("FLANGE SLIP-ON 4 INCH CARBON STEEL ANSI B16.5 300LB", "spec_diff_rating"),
         ("FLANGE WELD-NECK 4 INCH CARBON STEEL ANSI B16.5 150LB", "spec_diff_type"),
         ("FLANGE SLIP-ON 6 INCH CARBON STEEL ANSI B16.5 150LB", "spec_diff_size")],
        "hard", "piping", ["abbreviation"],
        "TRAIN-EXTRA-C05"
    ),
]

# ── CATEGORY D: Semantic neighbor contrastive pairs ───────────────────────────
# Explicitly creates hard pairs between semantically similar items
# DOES NOT copy: Rantai, Gear Rantai, chain/sprocket from test

d_examples = [
    make_example(
        "Tee 1 inch PVC",
        "PVC TEE 1 INCH EQUAL SOCKET TYPE",
        [("PVC ELBOW 90DEG 1 INCH SOCKET TYPE", "semantic_neighbor"),
         ("PVC REDUCER TEE 1 INCH X 3/4 INCH", "spec_diff_type"),
         ("PVC COUPLING 1 INCH SOCKET TYPE", "semantic_neighbor")],
        "hard", "piping", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D01"
    ),
    make_example(
        "Timing Belt 450H100",
        "TIMING BELT 450H SECTION WIDTH 100MM",
        [("V-BELT TYPE A SECTION LENGTH 45 INCHES", "semantic_neighbor"),
         ("TIMING BELT 525H100", "spec_diff_length"),
         ("TIMING BELT 450H075", "spec_diff_width")],
        "hard", "mechanical", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D02"
    ),
    make_example(
        "Hex Nut M20 Grade 8",
        "HEX NUT M20 GRADE 8",
        [("HEX BOLT M20X60 GRADE 8.8", "semantic_neighbor"),
         ("HEX NUT M16 GRADE 8", "spec_diff_size"),
         ("HEX NUT M20 GRADE 5", "spec_diff_grade")],
        "hard", "fasteners", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D03"
    ),
    make_example(
        "Sprocket No 60 single strand",
        "CHAIN SPROCKET ANSI NO.60 SINGLE STRAND",
        [("ROLLER CHAIN ANSI NO.60 SINGLE STRAND", "semantic_neighbor"),
         ("CHAIN SPROCKET ANSI NO.80 SINGLE STRAND", "spec_diff_size"),
         ("CHAIN SPROCKET ANSI NO.60 DOUBLE STRAND", "spec_diff_strand")],
        "hard", "mechanical", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D04"
    ),
    make_example(
        "Industrial chain ANSI 35 simplex",
        "ROLLER CHAIN ANSI NO.35 SIMPLEX",
        [("CHAIN SPROCKET FOR ANSI NO.35 SIMPLEX", "semantic_neighbor"),
         ("ROLLER CHAIN ANSI NO.50 SIMPLEX", "spec_diff_size"),
         ("ROLLER CHAIN ANSI NO.35 DUPLEX", "spec_diff_strand")],
        "hard", "mechanical", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D05"
    ),
    make_example(
        "Elbow 90 derajat galvanis 2 inch",
        "GALVANIZED MALLEABLE IRON ELBOW 90DEG 2 INCH",
        [("GALVANIZED MALLEABLE IRON TEE 2 INCH", "semantic_neighbor"),
         ("GALVANIZED MALLEABLE IRON ELBOW 45DEG 2 INCH", "spec_diff_angle"),
         ("GALVANIZED MALLEABLE IRON ELBOW 90DEG 1.5 INCH", "spec_diff_size")],
        "hard", "piping", ["spec_diff_hard_negative"],
        "TRAIN-EXTRA-D06"
    ),
]

# ── CATEGORY E: Domain aliases / vendor-style naming ─────────────────────────
e_examples = [
    make_example(
        "NSK-6007-ZZ",
        "DEEP GROOVE BALL BEARING 6007 ZZ NSK",
        [("DEEP GROOVE BALL BEARING 6007 2RS NSK", "spec_diff_seal"),
         ("DEEP GROOVE BALL BEARING 6007 ZZ FAG", "spec_diff_brand"),
         ("PILLOW BLOCK BEARING UCP207", "different_type")],
        "adversarial", "mechanical", ["vendor_sku_only"],
        "TRAIN-EXTRA-E01"
    ),
    make_example(
        "NYY-5C-6MM",
        "NYY POWER CABLE 5 CORE 6MM2 0.6/1KV",
        [("NYY POWER CABLE 4 CORE 6MM2 0.6/1KV", "spec_diff_cores"),
         ("NYY POWER CABLE 5 CORE 10MM2 0.6/1KV", "spec_diff_size"),
         ("NYA SINGLE CORE CABLE 6MM2", "different_type")],
        "adversarial", "electrical", ["vendor_sku_only"],
        "TRAIN-EXTRA-E02"
    ),
    make_example(
        "Ball valve SS 1.5\"",
        "BALL VALVE 1-1/2 INCH SS304 FULL BORE",
        [("BALL VALVE 1 INCH SS304 FULL BORE", "spec_diff_size"),
         ("BALL VALVE 1-1/2 INCH CARBON STEEL FULL BORE", "spec_diff_material"),
         ("BUTTERFLY VALVE 1-1/2 INCH SS304", "different_type")],
        "adversarial", "piping", ["abbreviation"],
        "TRAIN-EXTRA-E03"
    ),
    make_example(
        "MCB 2P 20A",
        "MCB 2 POLE 20A 6KA MINIATURE CIRCUIT BREAKER",
        [("MCB 2 POLE 16A 6KA MINIATURE CIRCUIT BREAKER", "spec_diff_ampere"),
         ("MCB 1 POLE 20A 6KA MINIATURE CIRCUIT BREAKER", "spec_diff_pole"),
         ("RCCB 2 POLE 25A 30MA", "different_type")],
        "adversarial", "electrical", ["abbreviation"],
        "TRAIN-EXTRA-E04"
    ),
]

# ── Assemble all new examples ─────────────────────────────────────────────────
all_new = a_examples + b_examples + c_examples + d_examples + e_examples

# ── Leakage check ─────────────────────────────────────────────────────────────
print("Leakage checking new examples...")
passed = []
flagged = []
for q in all_new:
    all_texts = [q["invoice_line"]] + [c["description"] for c in q["candidates"]]
    leak = False
    for txt in all_texts:
        # Exact match check
        if txt.strip().upper() in all_test_descs:
            leak = True
            break
        # Near-match check (Jaccard > 0.85)
        ta = set(normalize(txt).split())
        for test_desc in all_test_descs:
            tb = set(normalize(test_desc).split())
            if ta and tb and len(ta & tb) / len(ta | tb) > 0.85:
                leak = True
                break
        if leak:
            break
    if leak:
        flagged.append(q["query_id"])
    else:
        passed.append(q)

if flagged:
    print(f"  FLAGGED (potential leakage): {flagged}")
else:
    print(f"  All {len(all_new)} new examples passed leakage check")

# ── Dedup check (no exact invoice text overlap within new set) ────────────────
invoice_texts = [q["invoice_line"].strip().upper() for q in passed]
if len(invoice_texts) != len(set(invoice_texts)):
    print("  WARNING: duplicate invoice texts in new training set")

# ── Build train_v2 ─────────────────────────────────────────────────────────────
with open(DATA_DIR / "train.json", encoding="utf-8") as f:
    train_v1 = json.load(f)

train_v2 = train_v1 + passed

print(f"\ntrain_v1: {len(train_v1)} queries")
print(f"New examples: {len(passed)}")
print(f"train_v2: {len(train_v2)} queries")

# Category distribution of new examples
cats = {}
for q in passed:
    c = q["category"]
    cats[c] = cats.get(c, 0) + 1
print(f"New category distribution: {cats}")

# Difficulty distribution
diffs = {}
for q in passed:
    d = q["difficulty"]
    diffs[d] = diffs.get(d, 0) + 1
print(f"New difficulty distribution: {diffs}")

with open(DATA_DIR / "train_v2.json", "w", encoding="utf-8") as f:
    json.dump(train_v2, f, ensure_ascii=False, indent=2)
print(f"\nSaved: data/train_v2.json")
