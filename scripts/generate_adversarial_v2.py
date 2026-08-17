"""
Adversarial Benchmark v2 Generator
Target: 15-20 ranking queries across 6 adversarial categories.
Each query has 4-8 candidates.

Categories (per response-prompt §3):
  A. Specification traps       — one token/spec changes item identity
  B. Vendor SKU only           — invoice has only vendor code
  C. Near-identical descriptions — one token difference
  D. Candidate competition     — multiple near-identical PO lines
  E. Substitution              — plausible substitute not in PO
  F. Unmatched                 — no true PO counterpart

All items use TEST SPLIT materials only (no training set contamination).
Frozen before model evaluation.

Output: data/test_adversarial_v2.json
"""

import json
import random
from pathlib import Path

random.seed(99)  # deterministic, different from training seed

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── Test-split materials used (manually specified to avoid leakage) ────────────
# These must all be from catalog materials with split="test"
# Drawn from material_catalog.json test split

ADVERSARIAL_QUERIES = [

    # ── CATEGORY A: Specification Traps ──────────────────────────────────────

    {
        "query_id": "ADV-A-001",
        "category_type": "A_spec_trap",
        "description": "SS304 vs SS316 stainless grade confusion",
        "invoice_line": "Pipa Stainless Steel SS316 1 inch Sch 10",
        "po_line_id": "ADV-A-001-PL-C",
        "candidates": [
            {"po_line_id": "ADV-A-001-PL-C", "description": "STAINLESS STEEL PIPE SS316 1 INCH SCH 10",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-A-001-PL-N1", "description": "STAINLESS STEEL PIPE SS304 1 INCH SCH 10",
             "is_correct": False, "neg_type": "spec_diff_grade"},
            {"po_line_id": "ADV-A-001-PL-N2", "description": "STAINLESS STEEL PIPE SS316 2 INCH SCH 10",
             "is_correct": False, "neg_type": "spec_diff_size"},
            {"po_line_id": "ADV-A-001-PL-N3", "description": "STAINLESS STEEL PIPE SS304 2 INCH SCH 10",
             "is_correct": False, "neg_type": "spec_diff_grade_size"},
            {"po_line_id": "ADV-A-001-PL-N4", "description": "CARBON STEEL PIPE 1 INCH SCH 40",
             "is_correct": False, "neg_type": "different_material"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-A-002",
        "category_type": "A_spec_trap",
        "description": "Rebar D10 vs D12 diameter confusion",
        "invoice_line": "Besi Beton D12 ulir per batang",
        "po_line_id": "ADV-A-002-PL-C",
        "candidates": [
            {"po_line_id": "ADV-A-002-PL-C", "description": "DEFORMED STEEL BAR D12 GRADE BJTD40",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-A-002-PL-N1", "description": "DEFORMED STEEL BAR D10 GRADE BJTD40",
             "is_correct": False, "neg_type": "spec_diff_diameter"},
            {"po_line_id": "ADV-A-002-PL-N2", "description": "DEFORMED STEEL BAR D16 GRADE BJTD40",
             "is_correct": False, "neg_type": "spec_diff_diameter"},
            {"po_line_id": "ADV-A-002-PL-N3", "description": "PLAIN STEEL BAR D12 GRADE BJTP24",
             "is_correct": False, "neg_type": "spec_diff_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-A-003",
        "category_type": "A_spec_trap",
        "description": "Bearing seal type 2RS vs ZZ",
        "invoice_line": "Deep Groove Ball Bearing 6205 2RS",
        "po_line_id": "ADV-A-003-PL-C",
        "candidates": [
            {"po_line_id": "ADV-A-003-PL-C", "description": "DEEP GROOVE BALL BEARING 6205 2RS",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-A-003-PL-N1", "description": "DEEP GROOVE BALL BEARING 6205 ZZ",
             "is_correct": False, "neg_type": "spec_diff_seal"},
            {"po_line_id": "ADV-A-003-PL-N2", "description": "DEEP GROOVE BALL BEARING 6305 2RS",
             "is_correct": False, "neg_type": "spec_diff_size"},
            {"po_line_id": "ADV-A-003-PL-N3", "description": "DEEP GROOVE BALL BEARING 6205 OPEN",
             "is_correct": False, "neg_type": "spec_diff_seal"},
            {"po_line_id": "ADV-A-003-PL-N4", "description": "ANGULAR CONTACT BEARING 7205 2RS",
             "is_correct": False, "neg_type": "different_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-A-004",
        "category_type": "A_spec_trap",
        "description": "Cable 3x2.5 vs 4x2.5 core count",
        "invoice_line": "NYY Cable 4 core 2.5mm2 0.6/1kV",
        "po_line_id": "ADV-A-004-PL-C",
        "candidates": [
            {"po_line_id": "ADV-A-004-PL-C", "description": "NYY POWER CABLE 4 CORE 2.5MM2 0.6/1KV",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-A-004-PL-N1", "description": "NYY POWER CABLE 3 CORE 2.5MM2 0.6/1KV",
             "is_correct": False, "neg_type": "spec_diff_cores"},
            {"po_line_id": "ADV-A-004-PL-N2", "description": "NYY POWER CABLE 4 CORE 4MM2 0.6/1KV",
             "is_correct": False, "neg_type": "spec_diff_size"},
            {"po_line_id": "ADV-A-004-PL-N3", "description": "NYA SINGLE CORE CABLE 2.5MM2",
             "is_correct": False, "neg_type": "different_type"},
        ],
        "difficulty": "adversarial",
    },

    # ── CATEGORY B: Vendor SKU Only ───────────────────────────────────────────

    {
        "query_id": "ADV-B-001",
        "category_type": "B_vendor_sku",
        "description": "SKU-only invoice for MCB — no semantic content",
        "invoice_line": "MCB-3P-16A",
        "po_line_id": "ADV-B-001-PL-C",
        "candidates": [
            {"po_line_id": "ADV-B-001-PL-C", "description": "MCB 3 POLE 16A 6KA CIRCUIT BREAKER",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-B-001-PL-N1", "description": "MCB 3 POLE 32A 6KA CIRCUIT BREAKER",
             "is_correct": False, "neg_type": "spec_diff_ampere"},
            {"po_line_id": "ADV-B-001-PL-N2", "description": "MCB 1 POLE 16A 6KA CIRCUIT BREAKER",
             "is_correct": False, "neg_type": "spec_diff_pole"},
            {"po_line_id": "ADV-B-001-PL-N3", "description": "MCCB 3 POLE 100A 18KA",
             "is_correct": False, "neg_type": "different_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-B-002",
        "category_type": "B_vendor_sku",
        "description": "Part number only — bearing",
        "invoice_line": "SKF-6205-2RS",
        "po_line_id": "ADV-B-002-PL-C",
        "candidates": [
            {"po_line_id": "ADV-B-002-PL-C", "description": "DEEP GROOVE BALL BEARING 6205 2RS SKF",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-B-002-PL-N1", "description": "DEEP GROOVE BALL BEARING 6205 ZZ NSK",
             "is_correct": False, "neg_type": "spec_diff_seal_brand"},
            {"po_line_id": "ADV-B-002-PL-N2", "description": "PILLOW BLOCK BEARING UCP205 SKF",
             "is_correct": False, "neg_type": "different_type"},
            {"po_line_id": "ADV-B-002-PL-N3", "description": "CYLINDRICAL ROLLER BEARING NU205 SKF",
             "is_correct": False, "neg_type": "different_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-B-003",
        "category_type": "B_vendor_sku",
        "description": "Abbreviated material code — BRC mesh",
        "invoice_line": "BRC M6-150x150",
        "po_line_id": "ADV-B-003-PL-C",
        "candidates": [
            {"po_line_id": "ADV-B-003-PL-C", "description": "WELDED WIRE MESH BRC M6 150X150MM",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-B-003-PL-N1", "description": "WELDED WIRE MESH BRC M8 150X150MM",
             "is_correct": False, "neg_type": "spec_diff_wire"},
            {"po_line_id": "ADV-B-003-PL-N2", "description": "WELDED WIRE MESH BRC M6 200X200MM",
             "is_correct": False, "neg_type": "spec_diff_mesh"},
            {"po_line_id": "ADV-B-003-PL-N3", "description": "HEXAGONAL WIRE MESH 1 INCH",
             "is_correct": False, "neg_type": "different_type"},
            {"po_line_id": "ADV-B-003-PL-N4", "description": "BARBED WIRE GALVANIZED BWG 14",
             "is_correct": False, "neg_type": "different_item"},
        ],
        "difficulty": "adversarial",
    },

    # ── CATEGORY C: Near-Identical Descriptions ───────────────────────────────

    {
        "query_id": "ADV-C-001",
        "category_type": "C_near_identical",
        "description": "Oil viscosity grade SAE 40 vs SAE 15W-40",
        "invoice_line": "Engine oil SAE 15W-40 API CI-4 drum 209L",
        "po_line_id": "ADV-C-001-PL-C",
        "candidates": [
            {"po_line_id": "ADV-C-001-PL-C", "description": "DIESEL ENGINE OIL SAE 15W-40 API CI-4 DRUM 209L",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-C-001-PL-N1", "description": "DIESEL ENGINE OIL SAE 40 API CH-4 DRUM 209L",
             "is_correct": False, "neg_type": "spec_diff_grade"},
            {"po_line_id": "ADV-C-001-PL-N2", "description": "DIESEL ENGINE OIL SAE 15W-40 API CI-4 PAIL 20L",
             "is_correct": False, "neg_type": "spec_diff_packaging"},
            {"po_line_id": "ADV-C-001-PL-N3", "description": "GASOLINE ENGINE OIL SAE 15W-40 API SL DRUM 209L",
             "is_correct": False, "neg_type": "spec_diff_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-C-002",
        "category_type": "C_near_identical",
        "description": "Pipe schedule SCH 40 vs SCH 80",
        "invoice_line": "Carbon Steel Pipe 2 inch SCH 80 L=6m",
        "po_line_id": "ADV-C-002-PL-C",
        "candidates": [
            {"po_line_id": "ADV-C-002-PL-C", "description": "CARBON STEEL PIPE 2 INCH SCH 80 L=6M",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-C-002-PL-N1", "description": "CARBON STEEL PIPE 2 INCH SCH 40 L=6M",
             "is_correct": False, "neg_type": "spec_diff_schedule"},
            {"po_line_id": "ADV-C-002-PL-N2", "description": "CARBON STEEL PIPE 3 INCH SCH 80 L=6M",
             "is_correct": False, "neg_type": "spec_diff_size"},
            {"po_line_id": "ADV-C-002-PL-N3", "description": "GALVANIZED STEEL PIPE 2 INCH SCH 80 L=6M",
             "is_correct": False, "neg_type": "spec_diff_material"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-C-003",
        "category_type": "C_near_identical",
        "description": "Hex bolt grade 8.8 vs 10.9",
        "invoice_line": "Hex Bolt M16x60 Grade 10.9",
        "po_line_id": "ADV-C-003-PL-C",
        "candidates": [
            {"po_line_id": "ADV-C-003-PL-C", "description": "HEX BOLT M16X60 GRADE 10.9",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-C-003-PL-N1", "description": "HEX BOLT M16X60 GRADE 8.8",
             "is_correct": False, "neg_type": "spec_diff_grade"},
            {"po_line_id": "ADV-C-003-PL-N2", "description": "HEX BOLT M16X80 GRADE 10.9",
             "is_correct": False, "neg_type": "spec_diff_length"},
            {"po_line_id": "ADV-C-003-PL-N3", "description": "HEX BOLT M12X60 GRADE 10.9",
             "is_correct": False, "neg_type": "spec_diff_diameter"},
            {"po_line_id": "ADV-C-003-PL-N4", "description": "CARRIAGE BOLT M16X60 GRADE 8.8",
             "is_correct": False, "neg_type": "spec_diff_type_grade"},
        ],
        "difficulty": "adversarial",
    },

    # ── CATEGORY D: Candidate Competition ────────────────────────────────────

    {
        "query_id": "ADV-D-001",
        "category_type": "D_candidate_competition",
        "description": "Multiple similar thermal overload relays — must pick correct one",
        "invoice_line": "Thermal Overload Relay 4-6A Schneider",
        "po_line_id": "ADV-D-001-PL-C",
        "candidates": [
            {"po_line_id": "ADV-D-001-PL-C", "description": "THERMAL OVERLOAD RELAY 4-6A SCHNEIDER LRD08",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-D-001-PL-N1", "description": "THERMAL OVERLOAD RELAY 7-10A SCHNEIDER LRD12",
             "is_correct": False, "neg_type": "spec_diff_range"},
            {"po_line_id": "ADV-D-001-PL-N2", "description": "THERMAL OVERLOAD RELAY 4-6A CHINT",
             "is_correct": False, "neg_type": "spec_diff_brand"},
            {"po_line_id": "ADV-D-001-PL-N3", "description": "THERMAL OVERLOAD RELAY 2.5-4A SCHNEIDER LRD07",
             "is_correct": False, "neg_type": "spec_diff_range"},
            {"po_line_id": "ADV-D-001-PL-N4", "description": "BIMETAL OVERLOAD RELAY 4-6A 220V",
             "is_correct": False, "neg_type": "spec_diff_type"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-D-002",
        "category_type": "D_candidate_competition",
        "description": "Chain and sprocket — mutual competition",
        "invoice_line": "Roller Chain No. 40 duplex",
        "po_line_id": "ADV-D-002-PL-C",
        "candidates": [
            {"po_line_id": "ADV-D-002-PL-C", "description": "ROLLER CHAIN ANSI NO.40 DUPLEX",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-D-002-PL-N1", "description": "CHAIN SPROCKET FOR ANSI NO.40 DUPLEX",
             "is_correct": False, "neg_type": "related_item"},
            {"po_line_id": "ADV-D-002-PL-N2", "description": "ROLLER CHAIN ANSI NO.40 SIMPLEX",
             "is_correct": False, "neg_type": "spec_diff_strand"},
            {"po_line_id": "ADV-D-002-PL-N3", "description": "ROLLER CHAIN ANSI NO.50 DUPLEX",
             "is_correct": False, "neg_type": "spec_diff_size"},
        ],
        "difficulty": "adversarial",
    },

    # ── CATEGORY E: Substitution ──────────────────────────────────────────────

    {
        "query_id": "ADV-E-001",
        "category_type": "E_substitution",
        "description": "Invoice offers equivalent product not listed in PO",
        "invoice_line": "Hydraulic Hose SAE 100R2AT 1/2 inch 2000 PSI",
        "po_line_id": "ADV-E-001-PL-C",
        "candidates": [
            {"po_line_id": "ADV-E-001-PL-C", "description": "HYDRAULIC HOSE SAE 100R2AT 1/2 INCH",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-E-001-PL-N1", "description": "HYDRAULIC HOSE SAE 100R1AT 1/2 INCH",
             "is_correct": False, "neg_type": "spec_diff_type"},
            {"po_line_id": "ADV-E-001-PL-N2", "description": "HYDRAULIC HOSE SAE 100R2AT 3/4 INCH",
             "is_correct": False, "neg_type": "spec_diff_size"},
            {"po_line_id": "ADV-E-001-PL-N3", "description": "HYDRAULIC HOSE FITTING 1/2 INCH BSP MALE",
             "is_correct": False, "neg_type": "related_item"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-E-002",
        "category_type": "E_substitution",
        "description": "Invoice uses Indonesian term with plausible but wrong English match",
        "invoice_line": "Kunci Pas 17mm",
        "po_line_id": "ADV-E-002-PL-C",
        "candidates": [
            {"po_line_id": "ADV-E-002-PL-C", "description": "OPEN END WRENCH 17MM",
             "is_correct": True, "neg_type": None},
            {"po_line_id": "ADV-E-002-PL-N1", "description": "COMBINATION WRENCH 17MM",
             "is_correct": False, "neg_type": "plausible_substitute"},
            {"po_line_id": "ADV-E-002-PL-N2", "description": "BOX END WRENCH 17MM",
             "is_correct": False, "neg_type": "plausible_substitute"},
            {"po_line_id": "ADV-E-002-PL-N3", "description": "ADJUSTABLE WRENCH 12 INCH",
             "is_correct": False, "neg_type": "related_tool"},
        ],
        "difficulty": "adversarial",
    },

    # ── CATEGORY F: Unmatched ─────────────────────────────────────────────────

    {
        "query_id": "ADV-F-001",
        "category_type": "F_unmatched",
        "description": "Invoice line has no true PO counterpart",
        "invoice_line": "Safety Shoes Jogger Absolute 42",
        "po_line_id": None,   # no match
        "candidates": [
            {"po_line_id": "ADV-F-001-PL-N1", "description": "SAFETY HELMET FULL BRIM ANSI Z89.1",
             "is_correct": False, "neg_type": "different_ppe"},
            {"po_line_id": "ADV-F-001-PL-N2", "description": "SAFETY GOGGLES CLEAR LENS ANSI Z87.1",
             "is_correct": False, "neg_type": "different_ppe"},
            {"po_line_id": "ADV-F-001-PL-N3", "description": "WORK GLOVES NITRILE COATED SIZE L",
             "is_correct": False, "neg_type": "different_ppe"},
            {"po_line_id": "ADV-F-001-PL-N4", "description": "DUST MASK N95 PARTICULATE RESPIRATOR",
             "is_correct": False, "neg_type": "different_ppe"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-F-002",
        "category_type": "F_unmatched",
        "description": "Service charge on invoice — no material match",
        "invoice_line": "Biaya Pengiriman / Delivery Fee",
        "po_line_id": None,
        "candidates": [
            {"po_line_id": "ADV-F-002-PL-N1", "description": "PACKING MATERIAL CORRUGATED CARTON BOX",
             "is_correct": False, "neg_type": "different_item"},
            {"po_line_id": "ADV-F-002-PL-N2", "description": "STRETCH WRAP FILM 50CM",
             "is_correct": False, "neg_type": "different_item"},
            {"po_line_id": "ADV-F-002-PL-N3", "description": "FORKLIFT TRUCK 3 TON",
             "is_correct": False, "neg_type": "different_item"},
        ],
        "difficulty": "adversarial",
    },

    {
        "query_id": "ADV-F-003",
        "category_type": "F_unmatched",
        "description": "Wrong item number — item exists but PO has different spec",
        "invoice_line": "Mur Baut Stainless M8x20 SS316",
        "po_line_id": None,  # PO has M8x20 SS304, not SS316
        "candidates": [
            {"po_line_id": "ADV-F-003-PL-N1", "description": "HEX BOLT M8X20 STAINLESS STEEL SS304",
             "is_correct": False, "neg_type": "spec_diff_grade"},
            {"po_line_id": "ADV-F-003-PL-N2", "description": "HEX NUT M8 STAINLESS STEEL SS304",
             "is_correct": False, "neg_type": "different_item"},
            {"po_line_id": "ADV-F-003-PL-N3", "description": "HEX BOLT M10X25 STAINLESS STEEL SS316",
             "is_correct": False, "neg_type": "spec_diff_size"},
        ],
        "difficulty": "adversarial",
    },
]

# ── Shuffle candidates within each query ──────────────────────────────────────
def shuffle_candidates(queries):
    for q in queries:
        random.shuffle(q["candidates"])
    return queries

ADVERSARIAL_QUERIES = shuffle_candidates(ADVERSARIAL_QUERIES)

# ── Validate ──────────────────────────────────────────────────────────────────
def validate(queries):
    issues = []
    for q in queries:
        qid = q["query_id"]
        is_unmatched = q["po_line_id"] is None
        correct_count = sum(1 for c in q["candidates"] if c["is_correct"])
        if is_unmatched and correct_count != 0:
            issues.append(f"{qid}: unmatched but has {correct_count} correct candidates")
        if not is_unmatched and correct_count != 1:
            issues.append(f"{qid}: expected 1 correct, got {correct_count}")
        descs = [c["description"] for c in q["candidates"]]
        if len(descs) != len(set(descs)):
            issues.append(f"{qid}: duplicate candidate descriptions")
        if len(q["candidates"]) < 2:
            issues.append(f"{qid}: less than 2 candidates")
    return issues


issues = validate(ADVERSARIAL_QUERIES)
if issues:
    print("VALIDATION FAILURES:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("Validation PASSED")

# Stats
cat_counts = {}
for q in ADVERSARIAL_QUERIES:
    cat = q["category_type"].split("_")[0]
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

n_matched   = sum(1 for q in ADVERSARIAL_QUERIES if q["po_line_id"] is not None)
n_unmatched = sum(1 for q in ADVERSARIAL_QUERIES if q["po_line_id"] is None)

print(f"\nAdversarial benchmark v2: {len(ADVERSARIAL_QUERIES)} queries")
print(f"  Matched: {n_matched}, Unmatched: {n_unmatched}")
print(f"  Category distribution: {cat_counts}")
print(f"  Candidate range: {min(len(q['candidates']) for q in ADVERSARIAL_QUERIES)}"
      f"-{max(len(q['candidates']) for q in ADVERSARIAL_QUERIES)}")

out = DATA_DIR / "test_adversarial_v2.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(ADVERSARIAL_QUERIES, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out}")
print("\nNOTE: Benchmark frozen at this point. Evaluate models AFTER freezing.")
