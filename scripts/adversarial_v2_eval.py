"""
Adversarial v2 evaluation — per-category breakdown.
Evaluates lexical, pretrained, ft_v1, ft_v2 (best seed) on test_adversarial_v2.json.
Reports per ADV category (A=spec-trap, B=vendor-SKU, C=near-identical, 
D=candidate-competition, E=substitution, F=unmatched).
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "test_adversarial_v2.json", encoding="utf-8") as f:
    qs = json.load(f)

# ── Category definitions (from query ID prefix) ───────────────────────────────
CATEGORY_NAMES = {
    "ADV-A": "spec_trap",
    "ADV-B": "vendor_sku_only",
    "ADV-C": "near_identical",
    "ADV-D": "candidate_competition",
    "ADV-E": "substitution",
    "ADV-F": "unmatched",
}

def get_cat(qid):
    for prefix, name in CATEGORY_NAMES.items():
        if qid.startswith(prefix):
            return name
    return "unknown"

# ── Lexical ────────────────────────────────────────────────────────────────────
def normalize(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def jaccard(a, b):
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

# ── Semantic model loader ──────────────────────────────────────────────────────
def load_model(path):
    return SentenceTransformer(str(path))

def batch_encode_qs(model_obj, qs_list):
    all_inv   = list(set(q["invoice_line"] for q in qs_list))
    all_descs = list(set(c["description"] for q in qs_list for c in q["candidates"]))
    inv_arr  = model_obj.encode(all_inv,   convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    desc_arr = model_obj.encode(all_descs, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    return (
        {t: inv_arr[i]  for i, t in enumerate(all_inv)},
        {t: desc_arr[i] for i, t in enumerate(all_descs)},
    )

# ── Evaluators ─────────────────────────────────────────────────────────────────
def eval_lex(qs_list):
    results = {}
    for q in qs_list:
        cat = get_cat(q["query_id"])
        results.setdefault(cat, {"correct": 0, "total": 0, "matched": 0, "rank_sum": 0})
        matched = q.get("po_line_id") is not None
        if not matched:
            results[cat]["total"] += 1
            # For unmatched: model should ideally return low confidence; we count
            # the query as "handled" but can't measure top-1 accuracy.
            continue
        results[cat]["matched"] += 1
        results[cat]["total"] += 1
        inv = q["invoice_line"]
        cands = q["candidates"]
        scores = [(jaccard(inv, c["description"]), c["po_line_id"]) for c in cands]
        ranked = sorted(scores, key=lambda x: (-x[0], x[1]))
        correct_id = q["po_line_id"]
        for rank, (sc, pid) in enumerate(ranked, 1):
            if pid == correct_id:
                results[cat]["rank_sum"] += 1.0 / rank
                if rank == 1:
                    results[cat]["correct"] += 1
                break
    return results

def eval_semantic(qs_list, inv_embs, desc_embs):
    results = {}
    for q in qs_list:
        cat = get_cat(q["query_id"])
        results.setdefault(cat, {"correct": 0, "total": 0, "matched": 0, "rank_sum": 0})
        matched = q.get("po_line_id") is not None
        if not matched:
            results[cat]["total"] += 1
            continue
        results[cat]["matched"] += 1
        results[cat]["total"] += 1
        inv_e = inv_embs[q["invoice_line"]]
        cands = q["candidates"]
        scores = [(float(np.dot(inv_e, desc_embs[c["description"]])), c["po_line_id"]) for c in cands]
        ranked = sorted(scores, key=lambda x: (-x[0], x[1]))
        correct_id = q["po_line_id"]
        for rank, (sc, pid) in enumerate(ranked, 1):
            if pid == correct_id:
                results[cat]["rank_sum"] += 1.0 / rank
                if rank == 1:
                    results[cat]["correct"] += 1
                break
    return results

def format_cat_results(cat_results):
    out = {}
    for cat, r in sorted(cat_results.items()):
        m = r["matched"]
        c = r["correct"]
        out[cat] = {
            "top1":     round(c / m, 4) if m else None,
            "mrr":      round(r["rank_sum"] / m, 4) if m else None,
            "correct":  c,
            "matched":  m,
            "total":    r["total"],
        }
    # Aggregate over matched only
    tot_c = sum(r["correct"] for r in cat_results.values())
    tot_m = sum(r["matched"] for r in cat_results.values())
    tot_mrr = sum(r["rank_sum"] for r in cat_results.values())
    out["AGGREGATE_MATCHED"] = {
        "top1": round(tot_c / tot_m, 4) if tot_m else None,
        "mrr":  round(tot_mrr / tot_m, 4) if tot_m else None,
        "correct": tot_c, "matched": tot_m,
        "total": sum(r["total"] for r in cat_results.values()),
    }
    return out

# ── Run evaluations ───────────────────────────────────────────────────────────
print("="*70)
print("Adversarial v2 Evaluation — Per-Category Breakdown")
print("="*70)

# 1. Lexical
print("\n1. LEXICAL")
lex_cat = format_cat_results(eval_lex(qs))

# 2. Pretrained
print("2. PRETRAINED")
pretrained = load_model("paraphrase-multilingual-MiniLM-L12-v2")
inv_e, desc_e = batch_encode_qs(pretrained, qs)
pre_cat = format_cat_results(eval_semantic(qs, inv_e, desc_e))

# 3. FT-v1
ft_v1_path = Path("models/finetuned_minilm")
ft_v1_cat = None
if ft_v1_path.exists():
    print("3. FT-v1")
    ft1 = load_model(ft_v1_path)
    inv_e1, desc_e1 = batch_encode_qs(ft1, qs)
    ft_v1_cat = format_cat_results(eval_semantic(qs, inv_e1, desc_e1))

# 4. FT-v2 best seed
best_v2 = None
best_v2_mrr = -1
best_v2_label = None
for seed in [42, 43, 44]:
    lp = RESULTS_DIR / f"finetuned_v2_seed{seed}_training_log.json"
    mp = Path(f"models/finetuned_v2_seed{seed}")
    if lp.exists() and mp.exists():
        with open(lp) as f:
            log = json.load(f)
        entries = log.get("training_log", [])
        if entries:
            best_mrr = max(e["val_mrr@1"] for e in entries)
            if best_mrr > best_v2_mrr:
                best_v2_mrr = best_mrr
                best_v2 = mp
                best_v2_label = f"ft_v2_seed{seed}"

ft_v2_cat = None
if best_v2:
    print(f"4. FT-v2 ({best_v2_label})")
    ft2 = load_model(best_v2)
    inv_e2, desc_e2 = batch_encode_qs(ft2, qs)
    ft_v2_cat = format_cat_results(eval_semantic(qs, inv_e2, desc_e2))

# ── Print table ───────────────────────────────────────────────────────────────
all_cats = list(CATEGORY_NAMES.values()) + ["AGGREGATE_MATCHED"]
models_data = [
    ("lexical",        lex_cat),
    ("pretrained",     pre_cat),
]
if ft_v1_cat:  models_data.append(("ft_v1",    ft_v1_cat))
if ft_v2_cat:  models_data.append((best_v2_label or "ft_v2", ft_v2_cat))

print(f"\n{'Category':<28} {'n':>3}", end="")
for mname, _ in models_data:
    print(f"  {mname:>14}", end="")
print()
print("-"*80)
for cat in all_cats:
    n_matched = lex_cat.get(cat, {}).get("matched", 0)
    if n_matched == 0 and cat not in ["unmatched"]:
        # Show unmatched category separately
        pass
    row = f"  {cat:<26} {n_matched:>3}"
    for mname, mcat in models_data:
        val = mcat.get(cat, {}).get("top1")
        row += f"  {'N/A' if val is None else f'{val:.4f}':>14}"
    print(row)

# ── Save ───────────────────────────────────────────────────────────────────────
output = {
    "models": {m: d for m, d in models_data},
    "category_names": CATEGORY_NAMES,
    "best_ft_v2": best_v2_label,
    "n_queries": len(qs),
    "n_matched": sum(1 for q in qs if q.get("po_line_id")),
    "n_unmatched": sum(1 for q in qs if q.get("po_line_id") is None),
}
with open(RESULTS_DIR / "adversarial_v2_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/adversarial_v2_evaluation.json")
