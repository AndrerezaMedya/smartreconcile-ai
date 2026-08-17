"""
Phase 5: Fine-tuned Model Evaluation
Evaluates the fine-tuned model using the same metrics as baselines.
Also computes side-by-side comparison table: lexical vs pretrained vs finetuned.

Collapse detection (task-level, per plan §8.4):
  - Similarity variance across pairs
  - Positive/negative score separation
  - Retrieval degradation (vs pretrained baseline)
  - Rank collapse detection
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("models/finetuned_minilm")
DATA_FILES  = {
    "val":                  Path("data/val.json"),
    "test_synthetic_v1":    Path("data/test_synthetic_v1.json"),
    "test_synthetic_v2":    Path("data/test_synthetic_v2.json"),
    "test_hard_neg":        Path("data/test_hard_neg.json"),
    "test_adversarial_v2":  Path("data/test_adversarial_v2.json"),
}
RESULTS_DIR = Path("results")


def rank_candidates_semantic(query_emb, cand_embs):
    return cand_embs @ query_emb  # normalized embeddings


def compute_metrics(queries, model):
    by_difficulty = defaultdict(lambda: {
        "n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0,
        "recall3": 0, "scores_correct": [], "scores_incorrect": [],
        "n_unmatched": 0,
    })
    total = {"n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0, "recall3": 0}

    # Detect rank collapse: track if same candidate is always top-1
    top1_candidates = []

    all_invoice_lines = [q["invoice_line"] for q in queries]
    all_candidate_descs = []
    for q in queries:
        for c in q["candidates"]:
            all_candidate_descs.append(c["description"])

    invoice_embs = model.encode(all_invoice_lines, convert_to_numpy=True, normalize_embeddings=True)
    all_cand_embs = model.encode(all_candidate_descs, convert_to_numpy=True, normalize_embeddings=True)

    cand_idx = 0
    for qi, q in enumerate(queries):
        diff = q["difficulty"]
        is_unmatched = q.get("po_line_id") is None
        n_cands = len(q["candidates"])
        q_cand_embs = all_cand_embs[cand_idx:cand_idx + n_cands]
        cand_idx += n_cands

        if is_unmatched:
            by_difficulty[diff]["n_unmatched"] += 1
            continue

        scores = rank_candidates_semantic(invoice_embs[qi], q_cand_embs)
        ranked = sorted(zip(q["candidates"], scores.tolist()), key=lambda x: -x[1])
        top1_candidates.append(ranked[0][0]["po_line_id"])

        correct_id = q["po_line_id"]
        correct_rank = None
        for rank, (cand, score) in enumerate(ranked, 1):
            if cand["po_line_id"] == correct_id:
                correct_rank = rank
                by_difficulty[diff]["scores_correct"].append(score)
            else:
                by_difficulty[diff]["scores_incorrect"].append(score)

        if correct_rank is None:
            continue

        d = by_difficulty[diff]
        d["n"] += 1
        if correct_rank == 1: d["top1"] += 1; total["top1"] += 1
        if correct_rank <= 3:
            d["top3"] += 1; d["recall3"] += 1
            total["top3"] += 1; total["recall3"] += 1
        d["mrr_sum"] += 1.0 / correct_rank
        total["mrr_sum"] += 1.0 / correct_rank
        total["n"] += 1

    # Collapse detection
    all_scores = []
    for d in by_difficulty.values():
        all_scores.extend(d["scores_correct"])
        all_scores.extend(d["scores_incorrect"])

    collapse_flags = {}
    if all_scores:
        collapse_flags["score_variance"] = round(float(np.var(all_scores)), 6)
        collapse_flags["score_std"]      = round(float(np.std(all_scores)), 6)
        # Rank collapse: fraction of queries where same top-1 po_line_id appears
        if top1_candidates:
            from collections import Counter
            counts = Counter(top1_candidates)
            most_common_frac = counts.most_common(1)[0][1] / len(top1_candidates)
            collapse_flags["rank_collapse_fraction"] = round(most_common_frac, 4)
            collapse_flags["rank_collapse_detected"] = most_common_frac > 0.5
        # Low variance = possible collapse
        collapse_flags["low_variance_warning"] = collapse_flags["score_std"] < 0.05

    results = {}
    for diff, d in by_difficulty.items():
        n = d["n"]
        sc, si = d["scores_correct"], d["scores_incorrect"]
        results[diff] = {
            "n": n,
            "n_unmatched": d["n_unmatched"],
            "top1_acc":    round(d["top1"] / n, 4) if n else None,
            "top3_acc":    round(d["top3"] / n, 4) if n else None,
            "mrr":         round(d["mrr_sum"] / n, 4) if n else None,
            "recall_at_3": round(d["recall3"] / n, 4) if n else None,
            "score_correct_mean":   round(sum(sc)/len(sc), 4) if sc else None,
            "score_correct_min":    round(min(sc), 4) if sc else None,
            "score_incorrect_mean": round(sum(si)/len(si), 4) if si else None,
            "score_incorrect_max":  round(max(si), 4) if si else None,
            "separation_mean": round((sum(sc)/len(sc)) - (sum(si)/len(si)), 4) if sc and si else None,
        }

    n = total["n"]
    results["AGGREGATE"] = {
        "n": n,
        "top1_acc":    round(total["top1"] / n, 4) if n else None,
        "top3_acc":    round(total["top3"] / n, 4) if n else None,
        "mrr":         round(total["mrr_sum"] / n, 4) if n else None,
        "recall_at_3": round(total["recall3"] / n, 4) if n else None,
    }
    results["collapse_detection"] = collapse_flags
    return results


def print_comparison(ds_name, lexical, pretrained, finetuned):
    """Print a 3-way comparison table for Top-1 accuracy."""
    print(f"\n  {ds_name} — Top-1 Comparison")
    print(f"  {'Difficulty':<16} {'Lexical':>8}  {'Pretrained':>10}  {'Fine-tuned':>10}  {'Delta(F-L)':>10}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for diff in ["easy", "medium", "hard", "adversarial", "AGGREGATE"]:
        l = lexical.get(diff, {}).get("top1_acc")
        p = pretrained.get(diff, {}).get("top1_acc")
        f = finetuned.get(diff, {}).get("top1_acc")
        if l is None and p is None and f is None:
            continue
        ls = f"{l:.3f}" if l is not None else "n/a"
        ps = f"{p:.3f}" if p is not None else "n/a"
        fs = f"{f:.3f}" if f is not None else "n/a"
        delta = f"{f - l:+.3f}" if (f is not None and l is not None) else "n/a"
        print(f"  {diff:<16} {ls:>8}  {ps:>10}  {fs:>10}  {delta:>10}")


def main():
    print(f"Phase 5: Fine-tuned Model Evaluation")
    print(f"  Model: {MODEL_PATH}")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("ERROR: Fine-tuned model not found. Run finetune.py first.")
        return

    model = SentenceTransformer(str(MODEL_PATH))
    print(f"  Loaded fine-tuned model")

    # Load baseline results for comparison
    lex_results = {}
    pre_results = {}
    lex_path = RESULTS_DIR / "baseline_lexical.json"
    pre_path = RESULTS_DIR / "baseline_pretrained.json"
    if lex_path.exists():
        with open(lex_path, encoding="utf-8") as f:
            lex_results = json.load(f)
    if pre_path.exists():
        with open(pre_path, encoding="utf-8") as f:
            pre_results = json.load(f)

    all_results = {"model": str(MODEL_PATH)}

    for ds_name, path in DATA_FILES.items():
        with open(path, encoding="utf-8") as f:
            queries = json.load(f)

        print(f"\nDataset: {ds_name} ({len(queries)} queries)")
        metrics = compute_metrics(queries, model)
        all_results[ds_name] = metrics

        # Per-difficulty table
        print(f"  {'Difficulty':<16} {'N':>4}  {'Top-1':>6}  {'Top-3':>6}  {'MRR':>6}  {'R@3':>6}  {'Sep':>6}")
        print(f"  {'-'*16}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for diff in ["easy", "medium", "hard", "adversarial", "AGGREGATE"]:
            if diff not in metrics or diff == "collapse_detection":
                continue
            m = metrics[diff]
            if m.get("n", 0) == 0:
                continue
            top1 = f"{m['top1_acc']:.3f}" if m.get('top1_acc') is not None else "n/a"
            top3 = f"{m['top3_acc']:.3f}" if m.get('top3_acc') is not None else "n/a"
            mrr  = f"{m['mrr']:.3f}"      if m.get('mrr') is not None else "n/a"
            r3   = f"{m['recall_at_3']:.3f}" if m.get('recall_at_3') is not None else "n/a"
            sep  = f"{m['separation_mean']:.3f}" if m.get('separation_mean') is not None else "n/a"
            print(f"  {diff:<16} {m['n']:>4}  {top1:>6}  {top3:>6}  {mrr:>6}  {r3:>6}  {sep:>6}")

        # Collapse detection
        cd = metrics.get("collapse_detection", {})
        print(f"  Collapse detection: std={cd.get('score_std','?')}, "
              f"rank_collapse={cd.get('rank_collapse_detected','?')}, "
              f"low_var_warn={cd.get('low_variance_warning','?')}")

        # 3-way comparison
        if lex_results and pre_results:
            print_comparison(
                ds_name,
                lex_results.get(ds_name, {}),
                pre_results.get(ds_name, {}),
                metrics,
            )

    # H1 assessment
    print("\n" + "=" * 60)
    print("H1 ASSESSMENT: Does semantic add meaningful value?")
    hard_lex = lex_results.get("test_hard_neg", {}).get("hard", {}).get("top1_acc")
    hard_ft  = all_results.get("test_hard_neg", {}).get("hard", {}).get("top1_acc")
    if hard_lex is not None and hard_ft is not None:
        delta = hard_ft - hard_lex
        print(f"  Hard tier: Lexical={hard_lex:.3f}, Fine-tuned={hard_ft:.3f}, Delta={delta:+.3f}")
        if delta >= 0.10:
            print("  H1: STRONG GO (>= +10 pp)")
        elif delta > 0:
            print("  H1: GO — meaningful and consistent improvement")
        else:
            print("  H1: REVISE — no improvement on hard tier")

    # Save
    out_path = RESULTS_DIR / "finetuned_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
