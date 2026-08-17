"""
Phase 2: Lexical Baseline
Token-level Jaccard similarity on normalized descriptions.

Evaluation layers (per plan §4):
  Layer 2: Candidate ranking — Top-1, Top-3, MRR, Recall@3
  Layer 1: Pair similarity — score distribution stats

Reports per-difficulty, per-dataset, and aggregate.
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILES = {
    "val":                  Path("data/val.json"),
    "test_synthetic_v1":    Path("data/test_synthetic_v1.json"),
    "test_synthetic_v2":    Path("data/test_synthetic_v2.json"),
    "test_hard_neg":        Path("data/test_hard_neg.json"),
    "test_adversarial_v2":  Path("data/test_adversarial_v2.json"),
}
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def jaccard(a: str, b: str) -> float:
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Ranking metrics ───────────────────────────────────────────────────────────
def rank_candidates(query: str, candidates: list) -> list:
    """Return candidates sorted by descending Jaccard score.
    Stable tiebreak by po_line_id for determinism (CHECK-03 fix)."""
    scored = [(c, jaccard(query, c["description"])) for c in candidates]
    scored.sort(key=lambda x: (-x[1], x[0]["po_line_id"]))
    return scored


def compute_metrics(queries: list) -> dict:
    """
    Compute ranking metrics over a list of queries.
    Skips adversarial 'unmatched' queries (po_line_id is None) for ranking metrics.
    """
    by_difficulty = defaultdict(lambda: {
        "n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0,
        "recall3": 0, "scores_correct": [], "scores_incorrect": [],
        "n_unmatched": 0,
    })

    total = {"n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0, "recall3": 0}

    for q in queries:
        diff = q["difficulty"]
        is_unmatched = q.get("po_line_id") is None

        if is_unmatched:
            by_difficulty[diff]["n_unmatched"] += 1
            continue  # skip from ranking metrics

        correct_id = q["po_line_id"]
        ranked = rank_candidates(q["invoice_line"], q["candidates"])

        # Find rank of correct candidate (1-indexed)
        correct_rank = None
        for rank, (cand, score) in enumerate(ranked, 1):
            if cand["po_line_id"] == correct_id:
                correct_rank = rank
                by_difficulty[diff]["scores_correct"].append(score)
            else:
                by_difficulty[diff]["scores_incorrect"].append(score)

        if correct_rank is None:
            continue  # shouldn't happen, but guard

        d = by_difficulty[diff]
        d["n"] += 1
        if correct_rank == 1:
            d["top1"] += 1
            total["top1"] += 1
        if correct_rank <= 3:
            d["top3"] += 1
            d["recall3"] += 1
            total["recall3"] += 1
            total["top3"] += 1
        d["mrr_sum"] += 1.0 / correct_rank
        total["mrr_sum"] += 1.0 / correct_rank
        total["n"] += 1

    # Compile results
    results = {}
    for diff, d in by_difficulty.items():
        n = d["n"]
        if n == 0:
            results[diff] = {"n": 0, "top1_acc": None, "top3_acc": None,
                             "mrr": None, "recall_at_3": None,
                             "n_unmatched": d["n_unmatched"]}
            continue
        sc = d["scores_correct"]
        si = d["scores_incorrect"]
        results[diff] = {
            "n": n,
            "n_unmatched": d["n_unmatched"],
            "top1_acc": round(d["top1"] / n, 4),
            "top3_acc": round(d["top3"] / n, 4),
            "mrr": round(d["mrr_sum"] / n, 4),
            "recall_at_3": round(d["recall3"] / n, 4),
            "score_correct_mean": round(sum(sc)/len(sc), 4) if sc else None,
            "score_correct_min":  round(min(sc), 4) if sc else None,
            "score_incorrect_mean": round(sum(si)/len(si), 4) if si else None,
            "score_incorrect_max":  round(max(si), 4) if si else None,
            "separation_mean": round((sum(sc)/len(sc)) - (sum(si)/len(si)), 4) if sc and si else None,
        }

    n = total["n"]
    results["AGGREGATE"] = {
        "n": n,
        "top1_acc": round(total["top1"] / n, 4) if n else None,
        "top3_acc": round(total["top3"] / n, 4) if n else None,
        "mrr": round(total["mrr_sum"] / n, 4) if n else None,
        "recall_at_3": round(total["recall3"] / n, 4) if n else None,
    }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {"model": "lexical_jaccard"}

print("Phase 2: Lexical Baseline (Jaccard)")
print("=" * 60)

for ds_name, path in DATA_FILES.items():
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    metrics = compute_metrics(queries)
    all_results[ds_name] = metrics

    print(f"\nDataset: {ds_name} ({len(queries)} queries)")
    print(f"  {'Difficulty':<16} {'N':>4}  {'Top-1':>6}  {'Top-3':>6}  {'MRR':>6}  {'R@3':>6}  {'Sep':>6}")
    print(f"  {'-'*16}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for diff in ["easy", "medium", "hard", "adversarial", "AGGREGATE"]:
        if diff not in metrics:
            continue
        m = metrics[diff]
        if m["n"] == 0:
            print(f"  {diff:<16} {m['n']:>4}  {'n/a':>6}  {'n/a':>6}  {'n/a':>6}  {'n/a':>6}  {'n/a':>6}")
            continue
        top1 = f"{m['top1_acc']:.3f}" if m.get('top1_acc') is not None else "n/a"
        top3 = f"{m['top3_acc']:.3f}" if m.get('top3_acc') is not None else "n/a"
        mrr  = f"{m['mrr']:.3f}"      if m.get('mrr') is not None else "n/a"
        r3   = f"{m['recall_at_3']:.3f}" if m.get('recall_at_3') is not None else "n/a"
        sep  = f"{m['separation_mean']:.3f}" if m.get('separation_mean') is not None else "n/a"
        print(f"  {diff:<16} {m['n']:>4}  {top1:>6}  {top3:>6}  {mrr:>6}  {r3:>6}  {sep:>6}")

# Save
out_path = OUT_DIR / "baseline_lexical.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
