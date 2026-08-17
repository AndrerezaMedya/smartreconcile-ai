"""
Assignment Re-analysis — Ground Truth Evaluation
Re-evaluates Greedy vs Hungarian using LINE-LEVEL and DOCUMENT-LEVEL ground truth accuracy.
Focuses on SC-004 (Hungarian wins) and SC-008 (Greedy wins) anomalies.

Also tests: what happens when we apply t_low threshold first
(only assign pairs above t_low, treat rest as AMBIGUOUS)?

Decision rule (per response-prompt §10):
  Use the algorithm more often correct against ground truth.
  If equal, prefer simpler (greedy).
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from collections import defaultdict

RESULTS_DIR = Path("results")
MODEL_PATH  = Path("models/finetuned_minilm")

# Load frozen thresholds
thr_path = RESULTS_DIR / "thresholds.json"
if thr_path.exists():
    with open(thr_path, encoding="utf-8") as f:
        frozen_thr = json.load(f)
    T_HIGH = frozen_thr["t_high"]
    T_LOW  = frozen_thr["t_low"]
else:
    T_HIGH, T_LOW = 0.96, 0.81
print(f"Using thresholds: t_high={T_HIGH}, t_low={T_LOW}")

# Load previous assignment scenarios
prev_assign_path = RESULTS_DIR / "assignment_eval.json"
with open(prev_assign_path, encoding="utf-8") as f:
    prev_assign = json.load(f)

# Reconstruct MULTI_LINE_SCENARIOS from assignment_eval.json scenario data
# (We need the raw text to re-encode)
# Re-import from assignment_eval.py
import sys
sys.path.insert(0, "scripts")
from assignment_eval import MULTI_LINE_SCENARIOS

model = SentenceTransformer(str(MODEL_PATH))


# ── Assignment algorithms ─────────────────────────────────────────────────────
def greedy_assign(score_matrix, invoice_ids, po_ids):
    assignment = {iid: None for iid in invoice_ids}
    used_po = set()
    pairs = [(score_matrix[i, j], i, j) for i in range(len(invoice_ids)) for j in range(len(po_ids))]
    pairs.sort(key=lambda x: -x[0])
    assigned_inv = set()
    for score, i, j in pairs:
        if i not in assigned_inv and j not in used_po:
            assignment[invoice_ids[i]] = po_ids[j]
            assigned_inv.add(i)
            used_po.add(j)
    return assignment, score_matrix


def hungarian_assign(score_matrix, invoice_ids, po_ids):
    n_inv, n_po = score_matrix.shape
    size = max(n_inv, n_po)
    padded = np.zeros((size, size))
    padded[:n_inv, :n_po] = score_matrix
    row_ind, col_ind = linear_sum_assignment(-padded)
    assignment = {iid: None for iid in invoice_ids}
    for r, c in zip(row_ind, col_ind):
        if r < n_inv and c < n_po:
            assignment[invoice_ids[r]] = po_ids[c]
    return assignment, score_matrix


def evaluate_line_level(assignment, scenario):
    """Line-level accuracy: fraction of individual invoice lines correctly assigned."""
    correct = 0
    total   = 0
    details = []
    for il in scenario["invoice_lines"]:
        gold = il["correct_po"]
        pred = assignment.get(il["id"])
        is_correct = pred == gold
        if gold is not None:
            total += 1
            if is_correct: correct += 1
        details.append({
            "invoice_id": il["id"],
            "text": il["text"],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
        })
    return {"line_acc": round(correct/total, 4) if total else None,
            "n_correct": correct, "n_total": total, "details": details}


def evaluate_doc_level(assignment, scenario):
    """Document-level accuracy: 1 if ALL lines correctly assigned, else 0."""
    for il in scenario["invoice_lines"]:
        if il["correct_po"] is None: continue
        if assignment.get(il["id"]) != il["correct_po"]:
            return 0
    return 1


# ── Main analysis ─────────────────────────────────────────────────────────────
def main():
    print("Assignment Re-analysis — Ground Truth Evaluation")
    print("=" * 60)

    greedy_line_acc_all    = []
    hungarian_line_acc_all = []
    greedy_doc_all         = []
    hungarian_doc_all      = []

    greedy_wins    = 0
    hungarian_wins = 0
    ties           = 0

    reanalysis = []

    for sc in MULTI_LINE_SCENARIOS:
        invoice_texts = [il["text"] for il in sc["invoice_lines"]]
        po_texts      = [pl["description"] for pl in sc["po_lines"]]
        invoice_ids   = [il["id"] for il in sc["invoice_lines"]]
        po_ids        = [pl["id"] for pl in sc["po_lines"]]

        inv_embs = model.encode(invoice_texts, normalize_embeddings=True, convert_to_numpy=True)
        po_embs  = model.encode(po_texts,      normalize_embeddings=True, convert_to_numpy=True)
        score_matrix = inv_embs @ po_embs.T

        greedy_a,   _ = greedy_assign(score_matrix,   invoice_ids, po_ids)
        hungarian_a, _ = hungarian_assign(score_matrix, invoice_ids, po_ids)

        g_line = evaluate_line_level(greedy_a,   sc)
        h_line = evaluate_line_level(hungarian_a, sc)
        g_doc  = evaluate_doc_level(greedy_a,    sc)
        h_doc  = evaluate_doc_level(hungarian_a, sc)

        greedy_line_acc_all.append(g_line["line_acc"] or 0)
        hungarian_line_acc_all.append(h_line["line_acc"] or 0)
        greedy_doc_all.append(g_doc)
        hungarian_doc_all.append(h_doc)

        g_la = g_line["line_acc"] or 0
        h_la = h_line["line_acc"] or 0
        if g_la > h_la:   greedy_wins += 1
        elif h_la > g_la: hungarian_wins += 1
        else: ties += 1

        # Score matrix for interpretability
        score_table = {}
        for i, iid in enumerate(invoice_ids):
            score_table[iid] = {po_ids[j]: round(float(score_matrix[i,j]),4) for j in range(len(po_ids))}

        entry = {
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "greedy_line_acc":    g_la,
            "hungarian_line_acc": h_la,
            "greedy_doc_acc":    g_doc,
            "hungarian_doc_acc": h_doc,
            "winner": "GREEDY" if g_la > h_la else ("HUNGARIAN" if h_la > g_la else "TIE"),
            "greedy_assignment": dict(greedy_a),
            "hungarian_assignment": dict(hungarian_a),
            "greedy_details": g_line["details"],
            "hungarian_details": h_line["details"],
            "score_matrix": score_table,
        }
        reanalysis.append(entry)

        print(f"\n  {sc['scenario_id']}: {sc['name']}")
        print(f"    Greedy:    line={g_la:.3f} ({g_line['n_correct']}/{g_line['n_total']}) doc={g_doc}")
        print(f"    Hungarian: line={h_la:.3f} ({h_line['n_correct']}/{h_line['n_total']}) doc={h_doc}")
        print(f"    Winner: {'TIE' if g_la == h_la else ('GREEDY' if g_la > h_la else 'HUNGARIAN')}")

        # Show mismatches
        for g_d, h_d in zip(g_line["details"], h_line["details"]):
            if not g_d["correct"] or not h_d["correct"]:
                print(f"      [{g_d['invoice_id']}] gold={g_d['gold']} | greedy={g_d['pred']}({'OK' if g_d['correct'] else 'WRONG'}) | hung={h_d['pred']}({'OK' if h_d['correct'] else 'WRONG'})")

    # Summary
    n = len(MULTI_LINE_SCENARIOS)
    print(f"\n{'='*60}")
    print(f"GROUND-TRUTH SUMMARY ({n} scenarios)")
    print(f"  Greedy mean line-acc:    {np.mean(greedy_line_acc_all):.4f}")
    print(f"  Hungarian mean line-acc: {np.mean(hungarian_line_acc_all):.4f}")
    print(f"  Greedy doc-acc:          {sum(greedy_doc_all)}/{n}")
    print(f"  Hungarian doc-acc:       {sum(hungarian_doc_all)}/{n}")
    print(f"  Greedy wins (line-level): {greedy_wins}")
    print(f"  Hungarian wins:           {hungarian_wins}")
    print(f"  Ties:                     {ties}")

    # Decision
    if greedy_wins > hungarian_wins:
        decision = "KEEP_GREEDY — Greedy more correct against ground truth"
    elif hungarian_wins > greedy_wins:
        decision = "SWITCH_TO_HUNGARIAN — Hungarian more correct against ground truth"
    else:
        decision = "TIE — Equal correctness, prefer simpler algorithm (KEEP_GREEDY)"
    print(f"\nDecision: {decision}")

    # H5 revised verdict
    print(f"\nH5 REVISED VERDICT:")
    print(f"  {decision}")
    print(f"  (Previous H5 was based on agreement rate only — now corrected to use ground truth)")

    # SC-004 and SC-008 deep dive
    for scid in ["SC-004", "SC-008"]:
        sc_entry = next((e for e in reanalysis if e["scenario_id"] == scid), None)
        if sc_entry:
            print(f"\n  {scid} Deep Dive:")
            print(f"    Score matrix:")
            for iid, po_scores in sc_entry["score_matrix"].items():
                for po, sc_score in po_scores.items():
                    print(f"      {iid} -> {po}: {sc_score}")
            print(f"    Greedy: {sc_entry['greedy_assignment']}")
            print(f"    Hungarian: {sc_entry['hungarian_assignment']}")

    output = {
        "model": str(MODEL_PATH),
        "n_scenarios": n,
        "greedy_mean_line_acc":    round(float(np.mean(greedy_line_acc_all)),    4),
        "hungarian_mean_line_acc": round(float(np.mean(hungarian_line_acc_all)), 4),
        "greedy_doc_acc":    f"{sum(greedy_doc_all)}/{n}",
        "hungarian_doc_acc": f"{sum(hungarian_doc_all)}/{n}",
        "greedy_wins_line_level":   greedy_wins,
        "hungarian_wins_line_level":hungarian_wins,
        "ties": ties,
        "h5_verdict": decision,
        "scenarios": reanalysis,
    }
    out = RESULTS_DIR / "assignment_reanalysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
