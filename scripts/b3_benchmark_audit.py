"""
B3 — Benchmark Validity Audit
Classifies each of the 5 synthetic test failures and produces verdict.

Per Experiment B requirements:
  - classify: valid / generator_artifact / ambiguous / unrealistic
  - if artifact: create test_synthetic_v2.json (preserve v1)
  - document all findings
"""

import json
import shutil
from pathlib import Path

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load test_synthetic ───────────────────────────────────────────────────────
with open(DATA_DIR / "test_synthetic.json", encoding="utf-8") as f:
    test_v1 = json.load(f)

# ── Load catalog for context ──────────────────────────────────────────────────
with open(DATA_DIR / "material_catalog.json", encoding="utf-8") as f:
    catalog = json.load(f)
mat_by_id = {m["material_id"]: m for m in catalog["materials"]}

# ── B3 Audit Findings ─────────────────────────────────────────────────────────
# (Findings derived from manual inspection + generator source analysis)

FINDINGS = [
    {
        "query_id": "SYN-TE-0013",
        "difficulty": "medium",
        "category": "packaging",
        "material_id": "MAT-079",
        "invoice_line": "Stiker thermal varies",
        "correct_po": "ADHESIVE LABEL a",
        "variation_type": "synonym",
        "classification": "GENERATOR_ARTIFACT",
        "evidence": (
            "Generator line 242: invoice_desc = f\"{_pick(abbrevs)} {' '.join(list(spec.values())[:2])}\". "
            "MAT-079 has sizes=['varies'] in catalog as a legitimate placeholder (sizes vary by order). "
            "The generator picks abbreviation 'Stiker' + spec_values[:2] = 'thermal' + 'varies', "
            "producing 'Stiker thermal varies'. The word 'varies' is a catalog-level metadata placeholder, "
            "NOT a valid invoice description token. No real invoice would contain the word 'varies' "
            "as a product specification."
        ),
        "verdict": "INVALID — Remove from test_synthetic_v2. Document in v2 changelog.",
        "action": "REMOVE_FROM_V2",
        "fix_note": (
            "Fix generator: filter out spec values that are pure placeholders "
            "('varies', 'n/a', 'tbd', '-') when building synonym invoice descriptions."
        ),
    },
    {
        "query_id": "SYN-TE-0015",
        "difficulty": "medium",
        "category": "fasteners",
        "material_id": "MAT-081",
        "invoice_line": "Baut",
        "correct_po": "U-BOLT",
        "variation_type": "truncation",
        "classification": "AMBIGUOUS_LABEL",
        "evidence": (
            "MAT-081 canonical_name='Baut U / U-Bolt'. The generator truncation variation "
            "uses style_truncated() which returns just the first abbreviation: 'Baut'. "
            "'Baut' in Indonesian means 'bolt' in general — it refers to any bolt type. "
            "The correct PO is 'U-BOLT', which is a SPECIFIC subtype. "
            "A buyer writing only 'Baut' on an invoice almost certainly means a generic bolt, "
            "not specifically a U-Bolt. The label (U-BOLT as correct match) is technically correct "
            "per material ID but semantically ambiguous: 'Baut' alone does not unambiguously "
            "identify U-BOLT over 'BLIND RIVET' or other fasteners."
        ),
        "verdict": (
            "RETAIN with AMBIGUOUS flag. This represents a REAL and VALID difficulty: "
            "under-specified invoice lines are a genuine pain point in invoice-to-PO matching. "
            "However, it should be noted that lexical baseline also fails here, "
            "and the correct answer requires domain knowledge (MAT-081 context). "
            "Keep in v2 but flag as 'genuinely_ambiguous=True' in metadata."
        ),
        "action": "RETAIN_WITH_FLAG",
        "fix_note": "Add genuinely_ambiguous=True field. Keep in test_synthetic_v2.",
    },
    {
        "query_id": "SYN-TE-0022",
        "difficulty": "hard",
        "category": "structural_steel",
        "material_id": "MAT-070",
        "invoice_line": "BRC",
        "correct_po": "WELDED WIRE MESH",
        "variation_type": "spec_diff_hard_negative",
        "classification": "VALID_REAL_WORLD_DIFFICULTY",
        "evidence": (
            "MAT-070 canonical='Wire Mesh / Wiremesh', abbreviations=['Wiremesh', 'Wire Mesh', 'BRC']. "
            "'BRC' is a real-world industry abbreviation for BRC mesh (Bridon Ropes Company mesh), "
            "widely used in Indonesian construction industry to mean welded wire mesh. "
            "The abbreviation is listed in the material catalog as a valid abbreviation for MAT-070. "
            "The query is within the intended MVP scope: matching abbreviation-only invoice entries "
            "to their full PO descriptions is a primary use case. "
            "All three models fail (lexical, pretrained, fine-tuned) indicating this is a "
            "genuine challenge, not an evaluation flaw."
        ),
        "verdict": (
            "VALID — Retain in test_synthetic_v2. This is a legitimate hard abbreviation case "
            "within scope of the MVP. Generalized abbreviation training (not copying this exact case) "
            "is the correct response."
        ),
        "action": "RETAIN",
        "fix_note": "No fix needed. Generalized training abbreviation pairs for Experiment B1.",
    },
    {
        "query_id": "SYN-TE-0024",
        "difficulty": "hard",
        "category": "bearings",
        "material_id": "MAT-072",
        "invoice_line": "Rantai",
        "correct_po": "ROLLER CHAIN",
        "variation_type": "spec_diff_hard_negative",
        "classification": "VALID_SEMANTIC_NEIGHBOR_AMBIGUITY",
        "evidence": (
            "MAT-072='Rantai Industri' (industrial chain), abbreviations include 'Chain','Rantai','Roller Chain'. "
            "MAT-073='Sprocket', abbreviations include 'Sprocket','Gear Rantai'. "
            "Invoice 'Rantai' (general Indonesian for 'chain') could map to either a chain or a sprocket "
            "without additional context. However, the canonical correct match is ROLLER CHAIN (the chain itself), "
            "not the CHAIN SPROCKET (the gear that drives the chain). "
            "A supplier writing just 'Rantai' most likely means the chain, not the sprocket. "
            "The label is correct but the query is semantically underspecified. "
            "Models confuse this because 'Rantai' appears in both ROLLER CHAIN and CHAIN SPROCKET contexts "
            "in training."
        ),
        "verdict": (
            "VALID — Retain in test_synthetic_v2. Real-world difficulty: underspecified Indonesian terms "
            "with semantic neighbor confusion is a genuine challenge. The correct label assignment is "
            "defensible (chain=primary, sprocket=secondary component). "
            "No change needed; contrastive training examples are the response."
        ),
        "action": "RETAIN",
        "fix_note": "Create contrastive chain vs. sprocket training pairs (generalized, not this exact query).",
    },
    {
        "query_id": "SYN-TE-0025",
        "difficulty": "hard",
        "category": "bearings",
        "material_id": "MAT-073",
        "invoice_line": "Gear Rantai",
        "correct_po": "CHAIN SPROCKET",
        "variation_type": "spec_diff_hard_negative",
        "classification": "VALID_SEMANTIC_NEIGHBOR_AMBIGUITY",
        "evidence": (
            "MAT-073='Sprocket', abbreviations=['Sprocket','Gear Rantai']. "
            "'Gear Rantai' literally means 'chain gear' in Indonesian — a legitimate term for a sprocket. "
            "The correct match is CHAIN SPROCKET, not ROLLER CHAIN. "
            "Models confuse SYN-TE-0024 and SYN-TE-0025 symmetrically (assign each other's correct PO), "
            "indicating the embeddings place 'Rantai' and 'Gear Rantai' similarly relative to "
            "'ROLLER CHAIN' and 'CHAIN SPROCKET'. This is a real semantic neighbor ambiguity. "
            "The labels are correct. The confusion is a genuine model limitation."
        ),
        "verdict": (
            "VALID — Retain in test_synthetic_v2. Symmetric confusion with SYN-TE-0024 is expected "
            "and valid evidence of semantic neighbor limitation. Keep both."
        ),
        "action": "RETAIN",
        "fix_note": "Same as SYN-TE-0024: contrastive chain vs. sprocket training.",
    },
]

# ── Summary ───────────────────────────────────────────────────────────────────
n_total     = len(FINDINGS)
n_artifact  = sum(1 for f in FINDINGS if f["classification"] == "GENERATOR_ARTIFACT")
n_ambiguous = sum(1 for f in FINDINGS if "AMBIGUOUS" in f["classification"])
n_valid     = sum(1 for f in FINDINGS if f["classification"].startswith("VALID"))
n_remove    = sum(1 for f in FINDINGS if f["action"] == "REMOVE_FROM_V2")

print("B3 Benchmark Audit Results")
print("="*60)
for f in FINDINGS:
    print(f"\n  {f['query_id']} | {f['classification']}")
    print(f"  Action: {f['action']}")
    print(f"  Verdict: {f['verdict'][:80]}...")

print(f"\nSummary:")
print(f"  Total failures audited: {n_total}")
print(f"  Generator artifacts:    {n_artifact} (remove from v2)")
print(f"  Ambiguous labels:       {n_ambiguous} (retain with flag)")
print(f"  Valid difficulties:     {n_valid} (retain)")

# ── Create test_synthetic_v2.json ─────────────────────────────────────────────
# First, copy v1 for preservation
v1_path = DATA_DIR / "test_synthetic.json"
v1_copy = DATA_DIR / "test_synthetic_v1.json"
if not v1_copy.exists():
    shutil.copy2(v1_path, v1_copy)
    print(f"\nPreserved v1: {v1_copy}")
else:
    print(f"\nv1 already preserved: {v1_copy}")

remove_ids = {f["query_id"] for f in FINDINGS if f["action"] == "REMOVE_FROM_V2"}
retain_flag_ids = {f["query_id"] for f in FINDINGS if f["action"] == "RETAIN_WITH_FLAG"}

# Build v2
test_v2 = []
removed = []
for q in test_v1:
    if q["query_id"] in remove_ids:
        removed.append(q["query_id"])
        print(f"  Removed: {q['query_id']} ({q['invoice_line']}) — GENERATOR_ARTIFACT")
        continue
    q_copy = dict(q)
    if q["query_id"] in retain_flag_ids:
        q_copy["genuinely_ambiguous"] = True
        print(f"  Flagged: {q['query_id']} ({q['invoice_line']}) — AMBIGUOUS")
    test_v2.append(q_copy)

print(f"\ntest_synthetic_v1: {len(test_v1)} queries")
print(f"test_synthetic_v2: {len(test_v2)} queries ({len(removed)} removed)")

v2_path = DATA_DIR / "test_synthetic_v2.json"
with open(v2_path, "w", encoding="utf-8") as f:
    json.dump(test_v2, f, ensure_ascii=False, indent=2)
print(f"Saved: {v2_path}")

# ── Fix generator: filter placeholder spec values ─────────────────────────────
# Document the fix needed in generate_dataset.py
FIX_DESCRIPTION = {
    "file": "scripts/generate_dataset.py",
    "line": 242,
    "current": "invoice_desc = f\"{_pick(abbrevs)} {' '.join(list(spec.values())[:2])}\"",
    "fixed":   (
        "# Filter out placeholder values before building invoice desc\n"
        "PLACEHOLDER_VALUES = {'varies', 'n/a', 'tbd', '-', 'custom', 'variable'}\n"
        "spec_vals = [str(v) for v in list(spec.values())[:2] "
        "if str(v).lower() not in PLACEHOLDER_VALUES]\n"
        "invoice_desc = ' '.join([_pick(abbrevs)] + spec_vals)"
    ),
}
print(f"\nGenerator fix needed in {FIX_DESCRIPTION['file']} line {FIX_DESCRIPTION['line']}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
audit_output = {
    "total_failures_audited": n_total,
    "generator_artifacts": n_artifact,
    "ambiguous_labels": n_ambiguous,
    "valid_difficulties": n_valid,
    "removed_from_v2": list(remove_ids),
    "flagged_ambiguous": list(retain_flag_ids),
    "test_synthetic_v1_size": len(test_v1),
    "test_synthetic_v2_size": len(test_v2),
    "generator_fix": FIX_DESCRIPTION,
    "findings": FINDINGS,
}
with open(RESULTS_DIR / "benchmark_validity_v2.json", "w", encoding="utf-8") as f:
    json.dump(audit_output, f, ensure_ascii=False, indent=2)
print(f"Saved: results/benchmark_validity_v2.json")
