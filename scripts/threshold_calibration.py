"""
Phase 6: Threshold Calibration
Protocol per plan §11:
  1. Collect all similarity scores from validation set
  2. Plot correct vs incorrect score distributions
  3. Sweep threshold 0.50–0.95 in 0.01 steps
  4. For each threshold, compute Precision, Recall, FDR_high, Ambiguity rate, Unmatched rate
  5. Select thresholds: FDR_high <= 5% at MATCHED threshold
  6. Freeze thresholds BEFORE evaluating on final test set
  7. Save frozen thresholds to thresholds.json

Outputs:
  results/threshold_calibration.json  — full sweep results
  results/thresholds.json             — frozen thresholds to use in eval
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("results")
MODEL_FINETUNED = Path("models/finetuned_minilm")
MODEL_PRETRAINED = "paraphrase-multilingual-MiniLM-L12-v2"


def encode_dataset(model, queries):
    """Encode all invoice lines and candidates, return (correct_scores, incorrect_scores, all_scores)."""
    all_invoice_lines = [q["invoice_line"] for q in queries]
    all_cand_descs    = [c["description"] for q in queries for c in q["candidates"]]

    inv_embs  = model.encode(all_invoice_lines, normalize_embeddings=True, convert_to_numpy=True)
    cand_embs = model.encode(all_cand_descs,    normalize_embeddings=True, convert_to_numpy=True)

    correct_scores   = []
    incorrect_scores = []

    cand_idx = 0
    for qi, q in enumerate(queries):
        if q.get("po_line_id") is None:
            cand_idx += len(q["candidates"])
            continue
        correct_id = q["po_line_id"]
        for c in q["candidates"]:
            score = float(cand_embs[cand_idx] @ inv_embs[qi])
            if c["po_line_id"] == correct_id:
                correct_scores.append(score)
            else:
                incorrect_scores.append(score)
            cand_idx += 1

    return correct_scores, incorrect_scores


def sweep_thresholds(correct_scores, incorrect_scores, upper_range=(0.50, 0.96, 0.01)):
    """
    For each candidate MATCHED threshold t_high, compute:
      - Precision@high:  correct high-conf / all high-conf
      - FDR_high:        wrong high-conf / all high-conf  (safety metric, target <=5%)
      - MATCHED rate:    % of all pairs above t_high
      - Recall@high:     correct high-conf / all correct

    Returns list of sweep entries sorted by t_high.
    """
    all_scores = [(s, True)  for s in correct_scores] + \
                 [(s, False) for s in incorrect_scores]
    n_total   = len(all_scores)
    n_correct = len(correct_scores)

    sweep = []
    t = upper_range[0]
    while t <= upper_range[1]:
        t = round(t, 2)
        above = [(s, is_c) for s, is_c in all_scores if s >= t]
        n_above = len(above)
        n_above_correct = sum(1 for _, is_c in above if is_c)
        n_above_wrong   = n_above - n_above_correct

        precision = n_above_correct / n_above if n_above > 0 else None
        fdr_high  = n_above_wrong   / n_above if n_above > 0 else None
        recall    = n_above_correct / n_correct if n_correct > 0 else None
        matched_rate = n_above / n_total if n_total > 0 else None

        sweep.append({
            "threshold": t,
            "n_above":   n_above,
            "matched_rate": round(matched_rate, 4) if matched_rate is not None else None,
            "precision":    round(precision,    4) if precision    is not None else None,
            "fdr_high":     round(fdr_high,     4) if fdr_high     is not None else None,
            "recall":       round(recall,       4) if recall       is not None else None,
        })
        t += upper_range[2]

    return sweep


def select_thresholds(sweep: list, fdr_target: float = 0.05) -> dict:
    """
    Select t_high: highest threshold where FDR_high <= fdr_target AND matched_rate > 0.
    Select t_low:  threshold below which the model is effectively giving up (low confidence).
                   Set to t_high - 0.15 as a pragmatic starting point (AMBIGUOUS zone width).
    """
    # Candidates: FDR_high <= target AND at least some matches
    candidates = [
        e for e in sweep
        if e["fdr_high"] is not None
        and e["fdr_high"] <= fdr_target
        and e["matched_rate"] is not None
        and e["matched_rate"] > 0.0
    ]

    if not candidates:
        # No threshold achieves FDR target — use most conservative available
        candidates = sorted(sweep, key=lambda e: e["fdr_high"] or 1.0)
        t_high = candidates[0]["threshold"] if candidates else 0.90
    else:
        # Among valid candidates, prefer higher threshold (safer)
        t_high = max(e["threshold"] for e in candidates)

    t_low = round(max(0.40, t_high - 0.15), 2)

    # Compute stats at selected thresholds
    high_entry = next((e for e in sweep if e["threshold"] == t_high), {})
    low_entries = [e for e in sweep if e["threshold"] <= t_high]
    ambig_entries = [e for e in sweep if t_low <= e["threshold"] < t_high]

    return {
        "t_high": t_high,
        "t_low":  t_low,
        "at_t_high": high_entry,
        "fdr_target": fdr_target,
        "selection_note": f"t_high selected as highest threshold with FDR_high <= {fdr_target}",
        "t_low_note": "t_low = t_high - 0.15 (pragmatic AMBIGUOUS zone width; tune if needed)",
    }


def distribution_stats(scores: list, label: str) -> dict:
    if not scores:
        return {}
    arr = np.array(scores)
    return {
        "label": label,
        "n": len(scores),
        "mean":  round(float(arr.mean()),  4),
        "std":   round(float(arr.std()),   4),
        "min":   round(float(arr.min()),   4),
        "p25":   round(float(np.percentile(arr, 25)), 4),
        "p50":   round(float(np.percentile(arr, 50)), 4),
        "p75":   round(float(np.percentile(arr, 75)), 4),
        "max":   round(float(arr.max()),   4),
    }


def run_calibration(model, queries, model_label: str) -> dict:
    print(f"  Encoding {len(queries)} queries...")
    correct, incorrect = encode_dataset(model, queries)
    print(f"  Correct scores: {len(correct)}, Incorrect scores: {len(incorrect)}")

    sweep   = sweep_thresholds(correct, incorrect)
    thresholds = select_thresholds(sweep)
    c_stats = distribution_stats(correct,   "correct_matches")
    i_stats = distribution_stats(incorrect, "incorrect_matches")

    # Separation check
    sep = round(c_stats["mean"] - i_stats["mean"], 4) if c_stats and i_stats else None

    print(f"  Score separation (mean): {sep}")
    print(f"  Selected t_high={thresholds['t_high']}, t_low={thresholds['t_low']}")
    at_high = thresholds.get("at_t_high", {})
    print(f"  At t_high: FDR={at_high.get('fdr_high')}, Precision={at_high.get('precision')}, MATCHED_rate={at_high.get('matched_rate')}")

    return {
        "model": model_label,
        "score_stats": {"correct": c_stats, "incorrect": i_stats, "separation_mean": sep},
        "sweep": sweep,
        "selected_thresholds": thresholds,
    }


def main():
    print("Phase 6: Threshold Calibration")
    print("=" * 60)

    val_path = Path("data/val.json")
    with open(val_path, encoding="utf-8") as f:
        val_queries = json.load(f)

    results = {}

    # Calibrate on fine-tuned model (primary)
    if MODEL_FINETUNED.exists():
        print("\nModel: fine-tuned MiniLM")
        model = SentenceTransformer(str(MODEL_FINETUNED))
        results["finetuned"] = run_calibration(model, val_queries, "finetuned_minilm")
        frozen_thresholds = results["finetuned"]["selected_thresholds"]
    else:
        print("Fine-tuned model not found — using pretrained for calibration")
        model = SentenceTransformer(MODEL_PRETRAINED)
        results["pretrained"] = run_calibration(model, val_queries, MODEL_PRETRAINED)
        frozen_thresholds = results["pretrained"]["selected_thresholds"]

    # Save calibration results
    cal_path = RESULTS_DIR / "threshold_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nCalibration saved: {cal_path}")

    # Save frozen thresholds (used in all subsequent evaluation)
    thr_path = RESULTS_DIR / "thresholds.json"
    frozen = {
        "t_high": frozen_thresholds["t_high"],
        "t_low":  frozen_thresholds["t_low"],
        "fdr_target": frozen_thresholds["fdr_target"],
        "calibrated_on": "val.json",
        "note": "Thresholds FROZEN before test set evaluation. Do not tune on test set.",
    }
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    print(f"Frozen thresholds: {thr_path}")
    print(f"  t_high={frozen['t_high']}, t_low={frozen['t_low']}")

    print("\n[H3/H4] Threshold assessment:")
    at_high = frozen_thresholds.get("at_t_high", {})
    fdr = at_high.get("fdr_high")
    mr  = at_high.get("matched_rate")
    if fdr is not None and fdr <= 0.05:
        print(f"  H3: PASS — FDR_high={fdr} <= 5%")
    elif fdr is not None:
        print(f"  H3: REVISE — FDR_high={fdr} > 5%")
    else:
        print("  H3: Cannot assess")

    print(f"  H4: MATCHED rate at t_high = {mr} — report and assess workload reduction")


if __name__ == "__main__":
    main()
