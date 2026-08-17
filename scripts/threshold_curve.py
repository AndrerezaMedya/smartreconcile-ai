"""
Threshold Coverage Curve
Full sweep from 0.70 to 0.98 (step 0.02) for the fine-tuned model on val set.

For each threshold, records:
  - precision (high-conf correct / all high-conf)
  - fdr_high (high-conf wrong / all high-conf)
  - recall (high-conf correct / all correct)
  - coverage / MATCHED rate (pairs above threshold / all pairs)
  - ambiguity rate (pairs in [t_low, t_high))  — using t_low = threshold - 0.15
  - unmatched rate (pairs below t_low)

Also reports margin analysis:
  - margin = score(top1) - score(top2) per query
  - whether margin is a better uncertainty signal than raw cosine

Outputs:
  results/threshold_curve.json
  results/threshold_analysis.md
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("models/finetuned_minilm")

# ── Load val data and encode ───────────────────────────────────────────────────
val_path = Path("data/val.json")
with open(val_path, encoding="utf-8") as f:
    val_queries = json.load(f)

print("Loading fine-tuned model...")
model = SentenceTransformer(str(MODEL_PATH))

print("Encoding val set...")
all_invoice = [q["invoice_line"] for q in val_queries]
all_cands   = [c["description"] for q in val_queries for c in q["candidates"]]
inv_embs    = model.encode(all_invoice, normalize_embeddings=True, convert_to_numpy=True)
cand_embs   = model.encode(all_cands,  normalize_embeddings=True, convert_to_numpy=True)

# Collect per-pair scores
correct_scores   = []
incorrect_scores = []
per_query_ranked = []   # for margin analysis

cand_idx = 0
for qi, q in enumerate(val_queries):
    if q.get("po_line_id") is None:
        cand_idx += len(q["candidates"])
        continue
    correct_id = q["po_line_id"]
    ranked = []
    for c in q["candidates"]:
        score = float(cand_embs[cand_idx] @ inv_embs[qi])
        is_c = c["po_line_id"] == correct_id
        if is_c:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
        ranked.append((score, c["po_line_id"], is_c))
        cand_idx += 1
    ranked.sort(key=lambda x: -x[0])
    per_query_ranked.append({
        "query_id": q["query_id"],
        "difficulty": q["difficulty"],
        "correct_id": correct_id,
        "ranked": ranked,
    })

all_scores = [(s, True)  for s in correct_scores] + \
             [(s, False) for s in incorrect_scores]
n_total    = len(all_scores)
n_correct  = len(correct_scores)

print(f"  {len(per_query_ranked)} matched queries, {n_total} total pairs")
print(f"  Correct mean={np.mean(correct_scores):.4f}, Incorrect mean={np.mean(incorrect_scores):.4f}")


# ── Threshold sweep ────────────────────────────────────────────────────────────
print("\nSweeping thresholds...")

# Sweep from 0.50 to 0.98 in 0.02 steps (coarse), then 0.01 around key zones
thresholds = sorted(set(
    [round(x, 2) for x in np.arange(0.50, 0.99, 0.02)] +
    [round(x, 2) for x in np.arange(0.70, 0.92, 0.01)]
))

sweep_results = []
for t_high in thresholds:
    t_low  = round(max(0.40, t_high - 0.15), 2)
    above  = [(s, is_c) for s, is_c in all_scores if s >= t_high]
    ambig  = [(s, is_c) for s, is_c in all_scores if t_low <= s < t_high]
    below  = [(s, is_c) for s, is_c in all_scores if s < t_low]

    n_above  = len(above)
    n_ambig  = len(ambig)
    n_below  = len(below)

    n_above_correct = sum(1 for _, is_c in above if is_c)
    n_above_wrong   = n_above - n_above_correct

    precision    = n_above_correct / n_above  if n_above  > 0 else None
    fdr_high     = n_above_wrong   / n_above  if n_above  > 0 else None
    recall       = n_above_correct / n_correct if n_correct > 0 else None
    coverage     = n_above  / n_total           if n_total  > 0 else None
    ambig_rate   = n_ambig  / n_total           if n_total  > 0 else None
    unmatched_rt = n_below  / n_total           if n_total  > 0 else None

    sweep_results.append({
        "t_high": t_high,
        "t_low":  t_low,
        "n_above":       n_above,
        "n_ambig":       n_ambig,
        "n_below":       n_below,
        "precision":     round(precision,    4) if precision    is not None else None,
        "fdr_high":      round(fdr_high,     4) if fdr_high     is not None else None,
        "recall":        round(recall,       4) if recall       is not None else None,
        "coverage":      round(coverage,     4) if coverage     is not None else None,
        "ambig_rate":    round(ambig_rate,   4) if ambig_rate   is not None else None,
        "unmatched_rate":round(unmatched_rt, 4) if unmatched_rt is not None else None,
        "fdr_ok":        fdr_high is not None and fdr_high <= 0.05,
    })

# Find threshold with FDR <= 5% and maximum coverage (lowest t_high that satisfies safety)
fdr_ok_entries = [e for e in sweep_results if e.get("fdr_ok") and e["coverage"] is not None and e["coverage"] > 0]
if fdr_ok_entries:
    # Lowest threshold that satisfies FDR <= 5% -> most coverage
    best_t_high = min(fdr_ok_entries, key=lambda e: e["t_high"])
    print(f"\nOptimal t_high (max coverage with FDR<=5%): {best_t_high['t_high']}")
    print(f"  Coverage={best_t_high['coverage']}, FDR={best_t_high['fdr_high']}, Precision={best_t_high['precision']}, Recall={best_t_high['recall']}")
else:
    best_t_high = None
    print("\nNo threshold achieves FDR <= 5% with coverage > 0 — model confidence calibration issue")


# ── Margin analysis ────────────────────────────────────────────────────────────
print("\nMargin analysis (top1 - top2 score gap)...")

margin_stats = defaultdict(list)
margin_by_correctness = {"correct_top1": [], "wrong_top1": []}

for qr in per_query_ranked:
    ranked = qr["ranked"]
    if len(ranked) < 2:
        continue
    top1_score = ranked[0][0]
    top2_score = ranked[1][0]
    margin = top1_score - top2_score

    top1_correct = ranked[0][2]  # is_correct
    margin_stats[qr["difficulty"]].append(margin)
    key = "correct_top1" if top1_correct else "wrong_top1"
    margin_by_correctness[key].append(margin)

print(f"  Correct top1 margin: mean={np.mean(margin_by_correctness['correct_top1']):.4f} ± {np.std(margin_by_correctness['correct_top1']):.4f}")
print(f"  Wrong top1 margin:   mean={np.mean(margin_by_correctness['wrong_top1']):.4f} ± {np.std(margin_by_correctness['wrong_top1']):.4f}")

# Margin threshold sweep — does margin > X predict correct top1?
print("\n  Margin threshold sweep (for uncertainty detection):")
margin_sweep = []
for t_margin in [round(x, 2) for x in np.arange(0.00, 0.51, 0.05)]:
    high_margin = [qr for qr in per_query_ranked
                   if len(qr["ranked"]) >= 2 and (qr["ranked"][0][0] - qr["ranked"][1][0]) >= t_margin]
    if not high_margin:
        continue
    n_hm = len(high_margin)
    n_hm_correct = sum(1 for qr in high_margin if qr["ranked"][0][2])
    prec_margin = n_hm_correct / n_hm
    cov_margin  = n_hm / len(per_query_ranked)
    margin_sweep.append({
        "t_margin": t_margin,
        "n_queries": n_hm,
        "coverage":  round(cov_margin,  4),
        "precision": round(prec_margin, 4),
    })
    print(f"    margin≥{t_margin:.2f}: coverage={cov_margin:.3f}, precision={prec_margin:.3f}")


# ── Print key sweep points ─────────────────────────────────────────────────────
print("\nKey threshold sweep (t_high | FDR | Precision | Coverage | Recall):")
for e in sweep_results:
    if e["t_high"] in [0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        fdr_str = f"{e['fdr_high']:.3f}" if e['fdr_high'] is not None else "n/a"
        pre_str = f"{e['precision']:.3f}" if e['precision'] is not None else "n/a"
        cov_str = f"{e['coverage']:.3f}" if e['coverage'] is not None else "n/a"
        rec_str = f"{e['recall']:.3f}"   if e['recall']   is not None else "n/a"
        flag = " <-- FDR OK" if e.get("fdr_ok") else ""
        print(f"  t={e['t_high']:.2f}: FDR={fdr_str} Prec={pre_str} Cov={cov_str} Rec={rec_str}{flag}")


# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "model": str(MODEL_PATH),
    "calibrated_on": "val.json",
    "score_stats": {
        "correct_mean":   round(float(np.mean(correct_scores)), 4),
        "correct_std":    round(float(np.std(correct_scores)), 4),
        "incorrect_mean": round(float(np.mean(incorrect_scores)), 4),
        "incorrect_std":  round(float(np.std(incorrect_scores)), 4),
        "separation":     round(float(np.mean(correct_scores) - np.mean(incorrect_scores)), 4),
    },
    "optimal_t_high_fdr05": best_t_high,
    "threshold_sweep": sweep_results,
    "margin_analysis": {
        "correct_top1_mean": round(float(np.mean(margin_by_correctness["correct_top1"])), 4) if margin_by_correctness["correct_top1"] else None,
        "correct_top1_std":  round(float(np.std(margin_by_correctness["correct_top1"])),  4) if margin_by_correctness["correct_top1"] else None,
        "wrong_top1_mean":   round(float(np.mean(margin_by_correctness["wrong_top1"])),   4) if margin_by_correctness["wrong_top1"] else None,
        "wrong_top1_std":    round(float(np.std(margin_by_correctness["wrong_top1"])),    4) if margin_by_correctness["wrong_top1"] else None,
        "margin_sweep": margin_sweep,
    },
}
with open(RESULTS_DIR / "threshold_curve.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/threshold_curve.json")
