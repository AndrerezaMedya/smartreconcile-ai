"""
B2 — Hybrid Semantic + Lexical Matcher Evaluation

Tests: hybrid_score = alpha * semantic_score + (1 - alpha) * lexical_score
       with optional spec-aware bonus for exact token matches

Alpha is swept on val set only. Final evaluation on frozen test sets.

Evaluates: Lexical | Pretrained | FT-v1 | FT-v2 (best seed) | Hybrid
On: val | test_synthetic_v2 | test_hard_neg | test_adversarial_v2

Output:
  results/hybrid_evaluation.json
  results/hybrid_evaluation.md
  results/experiment_B_model_comparison.json
"""

import json
import re
import math
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PLACEHOLDER = {"varies", "n/a", "tbd", "-", "custom", "variable"}

# ── Lexical (Jaccard) ─────────────────────────────────────────────────────────
def normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def jaccard(a: str, b: str) -> float:
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# ── Spec-aware bonus ──────────────────────────────────────────────────────────
# Simple rule: bonus if any high-signal token (numbers, codes) exactly matches
SPEC_PATTERN = re.compile(r"\b(\d+[\w./\-]*|\w+\d+\w*)\b")

def spec_overlap_score(a: str, b: str) -> float:
    """Returns fraction of spec-tokens in query that also appear in candidate."""
    ta = set(SPEC_PATTERN.findall(normalize(a)))
    tb = set(SPEC_PATTERN.findall(normalize(b)))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)

# ── Stable sort Top-1 ─────────────────────────────────────────────────────────
def top1_stable(scores: list, candidates: list) -> dict:
    """Sort by score descending, break ties by po_line_id for determinism."""
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: (-x[0], x[1]["po_line_id"]),
    )
    return ranked[0][1]

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(qs, score_fn):
    """
    score_fn(invoice_line, candidate_description) -> float
    Returns: {top1, mrr, r3, n}
    """
    top1_correct = 0
    mrr_sum = 0.0
    r3_correct = 0
    matched = 0

    for q in qs:
        if q.get("po_line_id") is None:
            # Unmatched — no correct answer
            continue
        matched += 1
        inv = q["invoice_line"]
        cands = q["candidates"]
        correct_id = q["po_line_id"]

        scores = [score_fn(inv, c["description"]) for c in cands]
        ranked = sorted(
            zip(scores, cands),
            key=lambda x: (-x[0], x[1]["po_line_id"]),
        )

        for rank, (sc, c) in enumerate(ranked, 1):
            if c["po_line_id"] == correct_id:
                mrr_sum += 1.0 / rank
                if rank == 1:
                    top1_correct += 1
                if rank <= 3:
                    r3_correct += 1
                break

    n = matched
    return {
        "top1": top1_correct / n if n else 0.0,
        "mrr":  mrr_sum / n if n else 0.0,
        "r3":   r3_correct / n if n else 0.0,
        "n":    n,
    }

def compute_by_difficulty(qs, score_fn):
    by_diff = {}
    for q in qs:
        d = q.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(q)
    result = {}
    for d, dqs in by_diff.items():
        result[d] = compute_metrics(dqs, score_fn)
    result["AGGREGATE"] = compute_metrics(qs, score_fn)
    return result

# ── Dataset loader ─────────────────────────────────────────────────────────────
def load_dataset(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ── Load semantic models ──────────────────────────────────────────────────────
print("Loading models...")

pretrained = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("  Loaded pretrained")

ft_v1_path = Path("models/finetuned_minilm")
ft_v1 = SentenceTransformer(str(ft_v1_path)) if ft_v1_path.exists() else None
print(f"  FT-v1: {'loaded' if ft_v1 else 'NOT FOUND'}")

# Find best v2 seed (by val MRR from training log)
best_v2_model = None
best_v2_mrr   = -1.0
best_v2_seed  = None
for seed in [42, 43, 44]:
    log_path = RESULTS_DIR / f"finetuned_v2_seed{seed}_training_log.json"
    model_path = Path(f"models/finetuned_v2_seed{seed}")
    if log_path.exists() and model_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        entries = log.get("training_log", [])
        if entries:
            best_mrr = max(e["val_mrr@1"] for e in entries)
            print(f"  FT-v2 seed={seed}: best val MRR@1={best_mrr:.4f}")
            if best_mrr > best_v2_mrr:
                best_v2_mrr  = best_mrr
                best_v2_seed = seed
                best_v2_model = model_path

ft_v2 = SentenceTransformer(str(best_v2_model)) if best_v2_model else None
print(f"  FT-v2 best: seed={best_v2_seed}, MRR={best_v2_mrr:.4f}" if ft_v2 else "  FT-v2: NOT FOUND")

# ── Encode helper (caching) ────────────────────────────────────────────────────
def make_semantic_scorer(model_obj):
    """Returns a function (inv, desc) -> float using cosine similarity."""
    # Pre-encode is not ideal for the query-time eval pattern used here,
    # so we embed on the fly. For performance, batch-encode per dataset.
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    def scorer(inv: str, desc: str) -> float:
        emb_inv  = model_obj.encode([inv],  convert_to_numpy=True, normalize_embeddings=True)
        emb_desc = model_obj.encode([desc], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(emb_inv[0], emb_desc[0]))
    return scorer

# Batch scoring is much faster — rewrite to batch per query
def batch_score_dataset(model_obj, qs):
    """Returns {qid: {po_line_id: score}} for all matched queries."""
    import numpy as np
    # Collect all unique texts
    all_inv   = []
    all_descs = []
    for q in qs:
        all_inv.append(q["invoice_line"])
        for c in q["candidates"]:
            all_descs.append(c["description"])

    unique_inv   = list(set(all_inv))
    unique_descs = list(set(all_descs))

    print(f"    Encoding {len(unique_inv)} invoices + {len(unique_descs)} candidates...")
    emb_inv   = model_obj.encode(unique_inv,   convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    emb_descs = model_obj.encode(unique_descs, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)

    inv_map   = {t: emb_inv[i]   for i, t in enumerate(unique_inv)}
    desc_map  = {t: emb_descs[i] for i, t in enumerate(unique_descs)}

    result = {}
    for q in qs:
        qid = q["query_id"]
        inv_emb = inv_map[q["invoice_line"]]
        result[qid] = {}
        for c in q["candidates"]:
            desc_emb = desc_map[c["description"]]
            result[qid][c["po_line_id"]] = float(np.dot(inv_emb, desc_emb))
    return result

def compute_metrics_from_scores(qs, scores_dict):
    top1_correct = 0
    mrr_sum = 0.0
    r3_correct = 0
    matched = 0
    for q in qs:
        if q.get("po_line_id") is None:
            continue
        matched += 1
        qid = q["query_id"]
        correct_id = q["po_line_id"]
        cand_scores = [(scores_dict[qid][c["po_line_id"]], c["po_line_id"]) for c in q["candidates"]]
        ranked = sorted(cand_scores, key=lambda x: (-x[0], x[1]))
        for rank, (sc, pid) in enumerate(ranked, 1):
            if pid == correct_id:
                mrr_sum += 1.0 / rank
                if rank == 1:
                    top1_correct += 1
                if rank <= 3:
                    r3_correct += 1
                break
    n = matched
    return {
        "top1": round(top1_correct / n, 4) if n else 0.0,
        "mrr":  round(mrr_sum / n, 4)      if n else 0.0,
        "r3":   round(r3_correct / n, 4)   if n else 0.0,
        "n":    n,
    }

def compute_by_diff_from_scores(qs, scores_dict):
    by_diff = {}
    for q in qs:
        d = q.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(q)
    result = {}
    for d, dqs in sorted(by_diff.items()):
        result[d] = compute_metrics_from_scores(dqs, scores_dict)
    result["AGGREGATE"] = compute_metrics_from_scores(qs, scores_dict)
    return result

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "val":               DATA_DIR / "val.json",
    "test_synthetic_v1": DATA_DIR / "test_synthetic_v1.json",
    "test_synthetic_v2": DATA_DIR / "test_synthetic_v2.json",
    "test_hard_neg":     DATA_DIR / "test_hard_neg.json",
    "test_adversarial_v2": DATA_DIR / "test_adversarial_v2.json",
}
loaded = {name: load_dataset(p) for name, p in DATASETS.items() if p.exists()}

# ── Step 1: Alpha sweep on VAL only ──────────────────────────────────────────
# Sweep alpha in {0.0, 0.1, ..., 1.0} with spec bonus = {0.0, 0.05, 0.10}
print("\n=== STEP 1: Alpha sweep on val ===")

val_qs = loaded["val"]

# Precompute lexical scores for val
lex_scores_val = {}
for q in val_qs:
    qid = q["query_id"]
    lex_scores_val[qid] = {c["po_line_id"]: jaccard(q["invoice_line"], c["description"])
                            for c in q["candidates"]}

# Precompute spec-overlap for val
spec_scores_val = {}
for q in val_qs:
    qid = q["query_id"]
    spec_scores_val[qid] = {c["po_line_id"]: spec_overlap_score(q["invoice_line"], c["description"])
                             for c in q["candidates"]}

# Use best available semantic model for sweep
sweep_model = ft_v2 if ft_v2 else (ft_v1 if ft_v1 else pretrained)
print(f"  Sweeping with: {'ft_v2' if ft_v2 else ('ft_v1' if ft_v1 else 'pretrained')}")
sem_scores_val = batch_score_dataset(sweep_model, val_qs)

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPEC_BETAS = [0.0, 0.05, 0.10]

print(f"\n  {'alpha':>6} {'spec_beta':>10} {'val_top1':>10} {'val_mrr':>8}")
print("  " + "-"*40)

best_val_top1 = -1
best_alpha    = 0.5
best_beta     = 0.0

sweep_results = []
for alpha in ALPHAS:
    for beta in SPEC_BETAS:
        # hybrid = alpha * sem + (1-alpha) * lex + beta * spec_overlap
        def hybrid_score_val(qid, pid):
            s = alpha * sem_scores_val[qid].get(pid, 0.0)
            s += (1 - alpha) * lex_scores_val[qid].get(pid, 0.0)
            s += beta * spec_scores_val[qid].get(pid, 0.0)
            return s

        # Compute top-1 for val
        top1_c = 0
        mrr_s  = 0.0
        n      = 0
        for q in val_qs:
            if q.get("po_line_id") is None:
                continue
            n += 1
            qid = q["query_id"]
            correct_id = q["po_line_id"]
            ranked = sorted(
                [(hybrid_score_val(qid, c["po_line_id"]), c["po_line_id"]) for c in q["candidates"]],
                key=lambda x: (-x[0], x[1]),
            )
            for rank, (sc, pid) in enumerate(ranked, 1):
                if pid == correct_id:
                    mrr_s += 1.0 / rank
                    if rank == 1:
                        top1_c += 1
                    break

        t1  = top1_c / n if n else 0.0
        mrr = mrr_s  / n if n else 0.0
        sweep_results.append({"alpha": alpha, "spec_beta": beta, "val_top1": round(t1, 4), "val_mrr": round(mrr, 4)})

        if t1 > best_val_top1:
            best_val_top1 = t1
            best_alpha    = alpha
            best_beta     = beta

        print(f"  {alpha:>6.1f} {beta:>10.2f} {t1:>10.4f} {mrr:>8.4f}")

print(f"\n  Best alpha={best_alpha}, spec_beta={best_beta} → val_top1={best_val_top1:.4f}")

# ── Step 2: Evaluate all models on all test sets ──────────────────────────────
print("\n=== STEP 2: Full evaluation on all datasets ===")

def eval_all_models_on_dataset(name, qs):
    print(f"\nDataset: {name} ({len(qs)} queries)")
    results = {}

    # Lexical
    print("  Lexical...")
    lex_sc = {q["query_id"]: {c["po_line_id"]: jaccard(q["invoice_line"], c["description"])
                               for c in q["candidates"]} for q in qs}
    results["lexical"] = compute_by_diff_from_scores(qs, lex_sc)

    # Pretrained
    print("  Pretrained...")
    sem_sc = batch_score_dataset(pretrained, qs)
    results["pretrained"] = compute_by_diff_from_scores(qs, sem_sc)

    # FT-v1
    if ft_v1:
        print("  FT-v1...")
        ft1_sc = batch_score_dataset(ft_v1, qs)
        results["ft_v1"] = compute_by_diff_from_scores(qs, ft1_sc)

    # FT-v2
    if ft_v2:
        print("  FT-v2...")
        ft2_sc = batch_score_dataset(ft_v2, qs)
        results["ft_v2"] = compute_by_diff_from_scores(qs, ft2_sc)

        # Hybrid (best alpha from val sweep)
        print(f"  Hybrid (alpha={best_alpha}, spec_beta={best_beta})...")
        spec_sc = {q["query_id"]: {c["po_line_id"]: spec_overlap_score(q["invoice_line"], c["description"])
                                    for c in q["candidates"]} for q in qs}
        lex_sc2 = {q["query_id"]: {c["po_line_id"]: jaccard(q["invoice_line"], c["description"])
                                    for c in q["candidates"]} for q in qs}

        hybrid_sc = {}
        for q in qs:
            qid = q["query_id"]
            hybrid_sc[qid] = {}
            for c in q["candidates"]:
                pid = c["po_line_id"]
                hybrid_sc[qid][pid] = (
                    best_alpha       * ft2_sc[qid].get(pid, 0.0)
                    + (1-best_alpha) * lex_sc2[qid].get(pid, 0.0)
                    + best_beta      * spec_sc[qid].get(pid, 0.0)
                )
        results["hybrid"] = compute_by_diff_from_scores(qs, hybrid_sc)

    return results

all_results = {}
for dname, qs in loaded.items():
    all_results[dname] = eval_all_models_on_dataset(dname, qs)

# ── Print comparison table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("MODEL COMPARISON — Top-1 Accuracy")
print("="*70)
models_available = ["lexical", "pretrained", "ft_v1", "ft_v2", "hybrid"]

for dname, dresults in all_results.items():
    print(f"\nDataset: {dname}")
    # Header
    header = f"  {'Difficulty':20}"
    for m in models_available:
        if m in dresults:
            header += f"  {m:>12}"
    print(header)
    print("  " + "-"*70)
    # Rows
    diffs_order = ["easy", "medium", "hard", "adversarial", "AGGREGATE"]
    for diff in diffs_order:
        row = f"  {diff:20}"
        any_diff = False
        for m in models_available:
            if m in dresults and diff in dresults[m]:
                row += f"  {dresults[m][diff]['top1']:>12.4f}"
                any_diff = True
        if any_diff:
            print(row)

# ── NFR-09 check ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("NFR-09 CHECK (>=0.90 Top-1 on test_synthetic_v2)")
print("="*70)
if "test_synthetic_v2" in all_results:
    for m in models_available:
        if m in all_results["test_synthetic_v2"]:
            agg = all_results["test_synthetic_v2"][m].get("AGGREGATE", {})
            t1 = agg.get("top1", 0)
            status = "PASS" if t1 >= 0.90 else "FAIL"
            print(f"  {m:20}: top1={t1:.4f}  [{status}]")

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "best_alpha":      best_alpha,
    "best_spec_beta":  best_beta,
    "val_top1_at_best_alpha": best_val_top1,
    "ft_v2_best_seed": best_v2_seed,
    "sweep_results":   sweep_results,
    "results":         all_results,
}

with open(RESULTS_DIR / "hybrid_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/hybrid_evaluation.json")

# Also save full comparison
with open(RESULTS_DIR / "experiment_B_model_comparison.json", "w", encoding="utf-8") as f:
    json.dump({
        "models": models_available,
        "datasets": list(all_results.keys()),
        "results": all_results,
        "nfr09_target": 0.90,
        "alpha": best_alpha,
        "spec_beta": best_beta,
    }, f, ensure_ascii=False, indent=2)
print(f"Saved: results/experiment_B_model_comparison.json")
