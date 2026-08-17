"""
Phase 7: Assignment Algorithm Evaluation
Greedy vs Hungarian on multi-line invoice scenarios.

Per plan §10: build 10+ multi-line adversarial invoices and compare:
  - Greedy: sort all (invoice_line, po_line) pairs by score descending,
            assign first-come-first-served, each line used at most once
  - Hungarian: scipy.optimize.linear_sum_assignment on negated similarity matrix

Decision rule:
  - Greedy-Hungarian agreement > 95% → greedy is sufficient
  - Disagreement 5-10% AND Hungarian consistently better → switch to Hungarian
  - Disagreement > 10% → must switch to Hungarian
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("results")
MODEL_FINETUNED = Path("models/finetuned_minilm")
MODEL_PRETRAINED = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Multi-line test invoice scenarios ─────────────────────────────────────────
# Built from catalog materials — test split materials only
# Each scenario: list of (invoice_line_desc, correct_po_line_id, po_lines)
# po_lines: list of {id, description} — all candidate PO lines in this PO

MULTI_LINE_SCENARIOS = [
    {
        "scenario_id": "SC-001",
        "name": "Piping package — competing specs",
        "description": "Two invoice lines both want similar PO lines; greedy might assign wrong first",
        "invoice_lines": [
            {"id": "IL-1", "text": "Pipa Galvanis 2\" Sch.40 p.6m",      "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Galvanized Steel Pipe 3 inch Sch 40", "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Union Galv 1 inch",                    "correct_po": "PL-3"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "GALVANIZED STEEL PIPE 2 INCH SCH 40 L=6M"},
            {"id": "PL-2", "description": "GALVANIZED STEEL PIPE 3 INCH SCH 40 L=6M"},
            {"id": "PL-3", "description": "GALVANIZED UNION 1 INCH CLASS 150"},
            {"id": "PL-4", "description": "CARBON STEEL PIPE 2 INCH SCH 40 L=6M"},
        ],
    },
    {
        "scenario_id": "SC-002",
        "name": "Bearing package — size ambiguity",
        "description": "Multiple bearing lines where greedy may pick wrong sizes",
        "invoice_lines": [
            {"id": "IL-1", "text": "Bearing 6205 2RS SKF",   "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Laher 6305 ZZ",          "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Pillow Block UCP205",     "correct_po": "PL-3"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "DEEP GROOVE BALL BEARING 6205 2RS"},
            {"id": "PL-2", "description": "DEEP GROOVE BALL BEARING 6305 ZZ"},
            {"id": "PL-3", "description": "PILLOW BLOCK BEARING UCP205"},
            {"id": "PL-4", "description": "DEEP GROOVE BALL BEARING 6205 ZZ"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-003",
        "name": "Electrical package — competing ratings",
        "description": "MCB and MCCB lines with similar descriptions, different current ratings",
        "invoice_lines": [
            {"id": "IL-1", "text": "MCB 3P 16A Schneider",  "correct_po": "PL-1"},
            {"id": "IL-2", "text": "MCB 3P 32A",            "correct_po": "PL-2"},
            {"id": "IL-3", "text": "MCCB 3P 100A",          "correct_po": "PL-3"},
            {"id": "IL-4", "text": "Kontaktor 18A 220V AC", "correct_po": "PL-4"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "MCB 3P 16A 6KA"},
            {"id": "PL-2", "description": "MCB 3P 32A 6KA"},
            {"id": "PL-3", "description": "MCCB 3P 100A 18KA"},
            {"id": "PL-4", "description": "MAGNETIC CONTACTOR 18A COIL 220V AC"},
            {"id": "PL-5", "description": "MCB 3P 25A 6KA"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-004",
        "name": "Fasteners — size variants all in one PO",
        "description": "Multiple bolt sizes where each invoice line must claim its unique PO line",
        "invoice_lines": [
            {"id": "IL-1", "text": "Baut Hex M10x50 Grade 8.8",  "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Hex Bolt M12x60 Gr.8.8",     "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Mur M10",                    "correct_po": "PL-3"},
            {"id": "IL-4", "text": "Ring Plat M10",              "correct_po": "PL-4"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "HEX BOLT M10X50 GRADE 8.8"},
            {"id": "PL-2", "description": "HEX BOLT M12X60 GRADE 8.8"},
            {"id": "PL-3", "description": "HEX NUT M10 GRADE 8"},
            {"id": "PL-4", "description": "FLAT WASHER M10"},
            {"id": "PL-5", "description": "HEX BOLT M10X60 GRADE 8.8"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-005",
        "name": "Chemicals — oil types competing",
        "description": "Different oil types that share many tokens",
        "invoice_lines": [
            {"id": "IL-1", "text": "Oli Mesin Diesel SAE 15W-40 drum 209L", "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Hydraulic Oil ISO VG 46",                "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Gear Oil SAE 90 API GL-4",               "correct_po": "PL-3"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "DIESEL ENGINE OIL SAE 15W-40 API CI-4 DRUM 209L"},
            {"id": "PL-2", "description": "HYDRAULIC OIL ISO VG 46 DRUM 209L"},
            {"id": "PL-3", "description": "GEAR OIL SAE 90 API GL-4 PAIL 20L"},
            {"id": "PL-4", "description": "DIESEL ENGINE OIL SAE 40 API CH-4 DRUM 209L"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-006",
        "name": "Steel — structural profiles competing",
        "description": "H-beam and angle bar with similar token overlap",
        "invoice_lines": [
            {"id": "IL-1", "text": "H-Beam 150x150 SS400 6m",    "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Angle Bar 50x50x5mm L=6m",   "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Plat Baja 6mm 4x8 feet",     "correct_po": "PL-3"},
            {"id": "IL-4", "text": "Besi Kotak 40x40 t=2mm 6m",  "correct_po": "PL-4"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "H-BEAM 150X150 JIS G3101 SS400 L=6M"},
            {"id": "PL-2", "description": "ANGLE BAR 50X50X5MM JIS G3101 SS400 L=6M"},
            {"id": "PL-3", "description": "STEEL PLATE 6MM HOT ROLLED SS400 4X8FEET"},
            {"id": "PL-4", "description": "RECTANGULAR HOLLOW SECTION 40X40 T=2MM L=6M"},
            {"id": "PL-5", "description": "ANGLE BAR 65X65X6MM L=6M"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-007",
        "name": "Mixed package — one unmatched invoice line",
        "description": "Invoice has a line that has no corresponding PO line",
        "invoice_lines": [
            {"id": "IL-1", "text": "Safety Helmet putih MSA",     "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Work Gloves nitrile L",       "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Sepatu Safety Kings 42",      "correct_po": None},  # no match
        ],
        "po_lines": [
            {"id": "PL-1", "description": "SAFETY HELMET WHITE ANSI Z89.1"},
            {"id": "PL-2", "description": "NITRILE COATED WORK GLOVES SIZE L"},
            {"id": "PL-3", "description": "DUST MASK N95"},
        ],
    },
    {
        "scenario_id": "SC-008",
        "name": "Local-best vs global-best conflict",
        "description": "Greedy picks PL-1 for IL-1 but globally IL-2 should get PL-1",
        "invoice_lines": [
            {"id": "IL-1", "text": "Filter Oli Sakura diesel",    "correct_po": "PL-2"},
            {"id": "IL-2", "text": "Oil Filter diesel engine",    "correct_po": "PL-1"},
            {"id": "IL-3", "text": "Air Filter element kompressor", "correct_po": "PL-3"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "ENGINE OIL FILTER SPIN-ON"},
            {"id": "PL-2", "description": "OIL FILTER ELEMENT DIESEL"},
            {"id": "PL-3", "description": "AIR FILTER ELEMENT COMPRESSOR"},
        ],
    },
    {
        "scenario_id": "SC-009",
        "name": "Cable types — core/size combinations",
        "description": "Multiple cable types sharing many spec tokens",
        "invoice_lines": [
            {"id": "IL-1", "text": "Kabel NYY 4x2.5mm2 0.6/1kV",  "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Cable NYA 2.5mm2 merah",       "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Kabel NYAF 1.5mm2 serabut",    "correct_po": "PL-3"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "NYY POWER CABLE 4 CORE 2.5MM2 0.6/1KV"},
            {"id": "PL-2", "description": "NYA SINGLE CORE CABLE 2.5MM2"},
            {"id": "PL-3", "description": "NYAF FLEXIBLE CABLE 1.5MM2"},
            {"id": "PL-4", "description": "NYY POWER CABLE 3 CORE 2.5MM2 0.6/1KV"},  # distractor
        ],
    },
    {
        "scenario_id": "SC-010",
        "name": "Packaging items — 5 line invoice",
        "description": "Dense packaging invoice, several items with overlapping vocabulary",
        "invoice_lines": [
            {"id": "IL-1", "text": "Kardus double wall 40x60cm",   "correct_po": "PL-1"},
            {"id": "IL-2", "text": "Stretch Film 50cm 5kg roll",   "correct_po": "PL-2"},
            {"id": "IL-3", "text": "Lakban OPP 48mm coklat 90yd",  "correct_po": "PL-3"},
            {"id": "IL-4", "text": "PP Strapping Band 15mm",       "correct_po": "PL-4"},
            {"id": "IL-5", "text": "Bubble Wrap 25mm 125cm roll",  "correct_po": "PL-5"},
        ],
        "po_lines": [
            {"id": "PL-1", "description": "CORRUGATED CARTON BOX DOUBLE WALL"},
            {"id": "PL-2", "description": "STRETCH WRAP FILM 50CM 5KG ROLL"},
            {"id": "PL-3", "description": "OPP PACKING TAPE 48MM 90 YARD"},
            {"id": "PL-4", "description": "PP STRAPPING BAND 15MM"},
            {"id": "PL-5", "description": "BUBBLE WRAP 25MM BUBBLE 125CM"},
            {"id": "PL-6", "description": "OPP PACKING TAPE 48MM 100 YARD"},  # distractor
        ],
    },
]


# ── Assignment algorithms ─────────────────────────────────────────────────────
def greedy_assign(score_matrix: np.ndarray, invoice_ids: list, po_ids: list) -> dict:
    """
    Sort all pairs by score descending, assign first-come-first-served.
    Returns dict: invoice_id -> po_id (or None if unmatched)
    """
    n_inv, n_po = score_matrix.shape
    assignment = {iid: None for iid in invoice_ids}
    used_po = set()

    # All pairs sorted descending by score
    pairs = [(score_matrix[i, j], i, j) for i in range(n_inv) for j in range(n_po)]
    pairs.sort(key=lambda x: -x[0])

    assigned_inv = set()
    for score, i, j in pairs:
        if i not in assigned_inv and j not in used_po:
            assignment[invoice_ids[i]] = po_ids[j]
            assigned_inv.add(i)
            used_po.add(j)

    return assignment


def hungarian_assign(score_matrix: np.ndarray, invoice_ids: list, po_ids: list) -> dict:
    """
    scipy linear_sum_assignment on negated similarity matrix (maximize similarity).
    Returns dict: invoice_id -> po_id
    """
    # Pad to square if needed (add dummy rows/cols)
    n_inv, n_po = score_matrix.shape
    size = max(n_inv, n_po)
    padded = np.zeros((size, size))
    padded[:n_inv, :n_po] = score_matrix

    row_ind, col_ind = linear_sum_assignment(-padded)

    assignment = {iid: None for iid in invoice_ids}
    for r, c in zip(row_ind, col_ind):
        if r < n_inv and c < n_po:
            assignment[invoice_ids[r]] = po_ids[c]

    return assignment


def evaluate_assignment(assignment: dict, scenario: dict) -> dict:
    """Compute accuracy against ground truth."""
    correct = 0
    total = 0
    details = []
    for il in scenario["invoice_lines"]:
        pred = assignment.get(il["id"])
        gold = il["correct_po"]
        is_correct = pred == gold
        if gold is not None:  # skip unmatched ground truth
            total += 1
            if is_correct:
                correct += 1
        details.append({
            "invoice_id": il["id"],
            "invoice_text": il["text"],
            "gold_po": gold,
            "pred_po": pred,
            "correct": is_correct,
        })
    acc = correct / total if total > 0 else None
    return {"accuracy": acc, "correct": correct, "total": total, "details": details}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Phase 7: Assignment Algorithm Evaluation")
    print("=" * 60)

    model_path = str(MODEL_FINETUNED) if MODEL_FINETUNED.exists() else MODEL_PRETRAINED
    print(f"Model: {model_path}")
    model = SentenceTransformer(model_path)

    scenario_results = []
    greedy_total = 0
    hungarian_total = 0
    agreements = 0
    greedy_wins = 0
    hungarian_wins = 0
    tie_count = 0

    for sc in MULTI_LINE_SCENARIOS:
        invoice_texts = [il["text"] for il in sc["invoice_lines"]]
        po_texts      = [pl["description"] for pl in sc["po_lines"]]
        invoice_ids   = [il["id"] for il in sc["invoice_lines"]]
        po_ids        = [pl["id"] for pl in sc["po_lines"]]

        inv_embs = model.encode(invoice_texts, normalize_embeddings=True, convert_to_numpy=True)
        po_embs  = model.encode(po_texts,      normalize_embeddings=True, convert_to_numpy=True)

        score_matrix = inv_embs @ po_embs.T   # shape: (n_invoice, n_po)

        greedy_assign_result   = greedy_assign(score_matrix,   invoice_ids, po_ids)
        hungarian_assign_result = hungarian_assign(score_matrix, invoice_ids, po_ids)

        greedy_eval   = evaluate_assignment(greedy_assign_result,   sc)
        hungarian_eval = evaluate_assignment(hungarian_assign_result, sc)

        # Do they agree?
        agree = all(greedy_assign_result[iid] == hungarian_assign_result[iid]
                    for iid in invoice_ids)
        agreements += int(agree)

        if greedy_eval["accuracy"] is None or hungarian_eval["accuracy"] is None:
            tie_count += 1
        elif greedy_eval["accuracy"] > hungarian_eval["accuracy"]:
            greedy_wins += 1
        elif hungarian_eval["accuracy"] > greedy_eval["accuracy"]:
            hungarian_wins += 1
        else:
            tie_count += 1

        print(f"\n  {sc['scenario_id']}: {sc['name']}")
        print(f"    Greedy:    acc={greedy_eval['accuracy']}  ({greedy_eval['correct']}/{greedy_eval['total']})")
        print(f"    Hungarian: acc={hungarian_eval['accuracy']}  ({hungarian_eval['correct']}/{hungarian_eval['total']})")
        print(f"    Agreement: {'YES' if agree else 'NO'}")

        scenario_results.append({
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "greedy": greedy_eval,
            "hungarian": hungarian_eval,
            "assignment_agree": agree,
            "greedy_assignment": greedy_assign_result,
            "hungarian_assignment": hungarian_assign_result,
        })

    n_scenarios = len(MULTI_LINE_SCENARIOS)
    agreement_rate = agreements / n_scenarios

    print(f"\n{'='*60}")
    print(f"SUMMARY ({n_scenarios} scenarios)")
    print(f"  Agreement rate:  {agreement_rate:.2%}  ({agreements}/{n_scenarios})")
    print(f"  Greedy wins:     {greedy_wins}")
    print(f"  Hungarian wins:  {hungarian_wins}")
    print(f"  Ties:            {tie_count}")

    # H5 assessment
    print(f"\nH5 ASSESSMENT:")
    if agreement_rate > 0.95:
        verdict = "KEEP greedy — agreement > 95%"
    elif agreement_rate >= 0.90 and hungarian_wins > greedy_wins:
        verdict = "SWITCH to Hungarian — agreement 90-95% and Hungarian consistently better"
    elif agreement_rate < 0.90:
        verdict = "SWITCH to Hungarian — agreement < 90%"
    else:
        verdict = "KEEP greedy — agreement borderline but Hungarian not consistently better"
    print(f"  {verdict}")

    results = {
        "model": model_path,
        "n_scenarios": n_scenarios,
        "agreement_rate": round(agreement_rate, 4),
        "greedy_wins": greedy_wins,
        "hungarian_wins": hungarian_wins,
        "ties": tie_count,
        "h5_verdict": verdict,
        "scenarios": scenario_results,
    }

    out = RESULTS_DIR / "assignment_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
