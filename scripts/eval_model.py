"""
Evaluate a specific fine-tuned model (v1 or v2) on all datasets.
Usage:
  python scripts/eval_model.py --model models/finetuned_v2_seed42 --label ft_v2_seed42
"""

import argparse
import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--label", default=None)
args = parser.parse_args()

MODEL_PATH  = Path(args.model)
LABEL       = args.label or MODEL_PATH.name
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILES = {
    "val":                  DATA_DIR / "val.json",
    "test_synthetic_v1":    DATA_DIR / "test_synthetic_v1.json",
    "test_synthetic_v2":    DATA_DIR / "test_synthetic_v2.json",
    "test_hard_neg":        DATA_DIR / "test_hard_neg.json",
    "test_adversarial_v2":  DATA_DIR / "test_adversarial_v2.json",
}

print(f"Loading {MODEL_PATH}...")
model = SentenceTransformer(str(MODEL_PATH))
print(f"  dim: {model.get_embedding_dimension()}")

def encode_batch(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)

def compute_metrics(qs, inv_embs, desc_embs):
    """inv_embs: {invoice_line: emb}, desc_embs: {description: emb}"""
    top1 = mrr = r3 = 0
    margins = []
    n = 0
    for q in qs:
        if q.get("po_line_id") is None:
            continue
        n += 1
        inv_e = inv_embs[q["invoice_line"]]
        cands = q["candidates"]
        scs = [(float(np.dot(inv_e, desc_embs[c["description"]])), c["po_line_id"]) for c in cands]
        ranked = sorted(scs, key=lambda x: (-x[0], x[1]))
        top_scores = [r[0] for r in ranked]
        margin = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 1.0
        margins.append(margin)
        correct_id = q["po_line_id"]
        for rank, (sc, pid) in enumerate(ranked, 1):
            if pid == correct_id:
                mrr += 1.0 / rank
                if rank == 1: top1 += 1
                if rank <= 3: r3 += 1
                break
    if n == 0:
        return {"top1": 0, "mrr": 0, "r3": 0, "n": 0, "avg_margin": 0}
    return {
        "top1": round(top1/n, 4),
        "mrr":  round(mrr/n, 4),
        "r3":   round(r3/n, 4),
        "n": n,
        "avg_margin": round(float(np.mean(margins)), 4),
    }

def compute_by_difficulty(qs, inv_embs, desc_embs):
    by_diff = {}
    for q in qs:
        d = q.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(q)
    result = {}
    for d, dqs in sorted(by_diff.items()):
        result[d] = compute_metrics(dqs, inv_embs, desc_embs)
    result["AGGREGATE"] = compute_metrics(qs, inv_embs, desc_embs)
    return result

all_results = {}

print(f"\n{'='*60}")
print(f"Model: {LABEL}")
print(f"{'='*60}")

for dname, dpath in DATA_FILES.items():
    if not dpath.exists():
        print(f"  [{dname}] NOT FOUND — skip")
        continue
    with open(dpath, encoding="utf-8") as f:
        qs = json.load(f)
    print(f"\nDataset: {dname} ({len(qs)} queries)")

    # Batch encode
    all_inv   = list(set(q["invoice_line"] for q in qs))
    all_descs = list(set(c["description"] for q in qs for c in q["candidates"]))
    print(f"  Encoding {len(all_inv)} inv + {len(all_descs)} desc...")
    inv_emb_arr  = encode_batch(all_inv)
    desc_emb_arr = encode_batch(all_descs)
    inv_embs  = {t: inv_emb_arr[i]  for i, t in enumerate(all_inv)}
    desc_embs = {t: desc_emb_arr[i] for i, t in enumerate(all_descs)}

    res = compute_by_difficulty(qs, inv_embs, desc_embs)
    all_results[dname] = res

    # Print
    print(f"  {'Diff':15} {'N':>5} {'Top1':>7} {'MRR':>7} {'R@3':>7} {'Margin':>8}")
    print("  " + "-"*55)
    diffs_order = ["easy", "medium", "hard", "adversarial", "AGGREGATE"]
    for d in diffs_order:
        if d in res:
            r = res[d]
            print(f"  {d:15} {r['n']:>5} {r['top1']:>7.4f} {r['mrr']:>7.4f} {r['r3']:>7.4f} {r.get('avg_margin',0):>8.4f}")

# NFR-09
print(f"\n{'='*60}")
print("NFR-09 check (>=0.90 top1 on test_synthetic_v2)")
for dname in ["test_synthetic_v1", "test_synthetic_v2"]:
    if dname in all_results:
        agg = all_results[dname].get("AGGREGATE", {})
        t1 = agg.get("top1", 0)
        status = "PASS" if t1 >= 0.90 else "FAIL"
        print(f"  {dname}: {t1:.4f}  [{status}]")

# Save
out = {
    "model": str(MODEL_PATH),
    "label": LABEL,
    "results": all_results,
}
out_path = RESULTS_DIR / f"eval_{LABEL}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
