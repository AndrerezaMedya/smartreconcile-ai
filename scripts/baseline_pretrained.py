"""
Phase 3: Pretrained Semantic Baseline
Uses paraphrase-multilingual-MiniLM-L12-v2 (zero-shot, no fine-tuning).
Cosine similarity on sentence embeddings.

Evaluation identical to lexical baseline for direct comparison.
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_FILES = {
    "val":                  Path("data/val.json"),
    "test_synthetic_v1":    Path("data/test_synthetic_v1.json"),
    "test_synthetic_v2":    Path("data/test_synthetic_v2.json"),
    "test_hard_neg":        Path("data/test_hard_neg.json"),
    "test_adversarial_v2":  Path("data/test_adversarial_v2.json"),
}
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"  Max seq length: {model.max_seq_length}")
print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def encode(texts: list) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def rank_candidates(query_emb: np.ndarray, cand_embs: np.ndarray) -> np.ndarray:
    """Returns cosine similarity scores for each candidate."""
    # Embeddings are already normalized → dot product = cosine similarity
    return cand_embs @ query_emb


def compute_metrics(queries: list) -> dict:
    by_difficulty = defaultdict(lambda: {
        "n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0,
        "recall3": 0, "scores_correct": [], "scores_incorrect": [],
        "n_unmatched": 0,
    })
    total = {"n": 0, "top1": 0, "top3": 0, "mrr_sum": 0.0, "recall3": 0}

    # Batch encode all texts at once for efficiency
    all_invoice_lines = []
    all_candidate_descs = []
    for q in queries:
        all_invoice_lines.append(q["invoice_line"])
        for c in q["candidates"]:
            all_candidate_descs.append(c["description"])

    print(f"  Encoding {len(all_invoice_lines)} invoice lines + {len(all_candidate_descs)} candidates...")
    invoice_embs = encode(all_invoice_lines)

    # Encode candidates in one batch
    all_cand_embs = encode(all_candidate_descs)

    # Split candidate embeddings back per query
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

        scores = rank_candidates(invoice_embs[qi], q_cand_embs)
        ranked = sorted(zip(q["candidates"], scores.tolist()), key=lambda x: -x[1])

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
        if correct_rank == 1:
            d["top1"] += 1; total["top1"] += 1
        if correct_rank <= 3:
            d["top3"] += 1; d["recall3"] += 1
            total["top3"] += 1; total["recall3"] += 1
        d["mrr_sum"] += 1.0 / correct_rank
        total["mrr_sum"] += 1.0 / correct_rank
        total["n"] += 1

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
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {"model": MODEL_NAME}

print("\nPhase 3: Pretrained Semantic Baseline")
print("=" * 60)

for ds_name, path in DATA_FILES.items():
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\nDataset: {ds_name} ({len(queries)} queries)")
    metrics = compute_metrics(queries)
    all_results[ds_name] = metrics

    print(f"  {'Difficulty':<16} {'N':>4}  {'Top-1':>6}  {'Top-3':>6}  {'MRR':>6}  {'R@3':>6}  {'Sep':>6}")
    print(f"  {'-'*16}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for diff in ["easy", "medium", "hard", "adversarial", "AGGREGATE"]:
        if diff not in metrics:
            continue
        m = metrics[diff]
        if m["n"] == 0:
            print(f"  {diff:<16} {m['n']:>4}  {'n/a':>6}")
            continue
        top1 = f"{m['top1_acc']:.3f}" if m.get('top1_acc') is not None else "n/a"
        top3 = f"{m['top3_acc']:.3f}" if m.get('top3_acc') is not None else "n/a"
        mrr  = f"{m['mrr']:.3f}"      if m.get('mrr') is not None else "n/a"
        r3   = f"{m['recall_at_3']:.3f}" if m.get('recall_at_3') is not None else "n/a"
        sep  = f"{m['separation_mean']:.3f}" if m.get('separation_mean') is not None else "n/a"
        print(f"  {diff:<16} {m['n']:>4}  {top1:>6}  {top3:>6}  {mrr:>6}  {r3:>6}  {sep:>6}")

out_path = OUT_DIR / "baseline_pretrained.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
