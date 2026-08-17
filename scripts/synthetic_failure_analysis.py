"""
Synthetic Failure Analysis
Investigates why fine-tuned model achieves only 75% Top-1 on synthetic test set.

For every failed query, classifies the failure into:
  - LEXICAL_DOMINANCE: lexical baseline also fails → inherently ambiguous text
  - PRETRAINED_BETTER: pretrained correct, finetuned wrong → fine-tuning regression
  - SEMANTIC_FAILURE: both pretrained and finetuned fail → embedding limitation
  - HARD_NEG_CONFUSION: predicted candidate is a spec_diff hard negative
  - LABEL_ISSUE: ground truth or candidate construction anomaly
  - CANDIDATE_ISSUE: candidate set may not reflect realistic options
  - UNREALISTIC: synthetic example appears unrealistic

Also analyzes:
  - Fine-tuning per-difficulty benefit breakdown
  - Score margin for correct vs wrong predictions
  - Pattern of which candidate was incorrectly ranked #1

Outputs:
  results/synthetic_failure_analysis.md
  results/synthetic_failure_cases.json
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict

RESULTS_DIR = Path("results")
MODEL_FT    = Path("models/finetuned_minilm")
MODEL_PRE   = "paraphrase-multilingual-MiniLM-L12-v2"


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def jaccard(a, b):
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def encode_and_rank(model, queries):
    """Returns {query_id: {'top1': po_line_id, 'rank_of_correct': int, 'scores': {...}}}"""
    all_inv   = [q["invoice_line"] for q in queries]
    all_cands = [c["description"] for q in queries for c in q["candidates"]]
    inv_embs  = model.encode(all_inv,   normalize_embeddings=True, convert_to_numpy=True)
    cand_embs = model.encode(all_cands, normalize_embeddings=True, convert_to_numpy=True)

    results = {}
    cand_idx = 0
    for qi, q in enumerate(queries):
        if q.get("po_line_id") is None:
            cand_idx += len(q["candidates"])
            continue
        correct_id = q["po_line_id"]
        scored = []
        for c in q["candidates"]:
            score = float(cand_embs[cand_idx] @ inv_embs[qi])
            scored.append((c["po_line_id"], score, c.get("neg_type", ""), c["is_correct"]))
            cand_idx += 1
        scored.sort(key=lambda x: -x[1])
        correct_rank = next((i+1 for i, (pid,_,_,_) in enumerate(scored) if pid == correct_id), None)
        results[q["query_id"]] = {
            "top1":              scored[0][0],
            "top1_correct":      scored[0][0] == correct_id,
            "rank_of_correct":   correct_rank,
            "top1_neg_type":     scored[0][2] if not scored[0][3] else "correct",
            "top1_score":        round(scored[0][1], 4),
            "top2_score":        round(scored[1][1], 4) if len(scored) > 1 else None,
            "correct_score":     round(next((s for pid,s,_,_ in scored if pid==correct_id), 0.0), 4),
            "margin":            round(scored[0][1] - scored[1][1], 4) if len(scored)>1 else None,
            "full_ranked":       [(pid, round(s,4), nt, ic) for pid,s,nt,ic in scored],
        }
    return results


def lexical_rank(queries):
    results = {}
    for q in queries:
        if q.get("po_line_id") is None: continue
        correct_id = q["po_line_id"]
        scored = [(c["po_line_id"], jaccard(q["invoice_line"], c["description"])) for c in q["candidates"]]
        scored.sort(key=lambda x: -x[1])
        results[q["query_id"]] = {
            "top1": scored[0][0],
            "top1_correct": scored[0][0] == correct_id,
        }
    return results


def classify_failure(q, lex_result, pre_result, ft_result):
    """Classify why fine-tuned model failed on this query."""
    reasons = []
    detail = {}

    ft_wrong  = not ft_result.get("top1_correct", True)
    pre_wrong = not pre_result.get("top1_correct", True)
    lex_wrong = not lex_result.get("top1_correct", True)

    if not ft_wrong:
        return "CORRECT", {}

    # Primary classification
    if lex_wrong and pre_wrong and ft_wrong:
        reasons.append("SEMANTIC_FAILURE")   # All models fail — inherently hard
    elif not lex_wrong and ft_wrong:
        reasons.append("FINE_TUNE_REGRESSION")  # Lexical wins, finetuned worse
    elif not pre_wrong and ft_wrong:
        reasons.append("FINE_TUNE_REGRESSION")  # Pretrained wins, finetuned worse
    elif pre_wrong and ft_wrong and not lex_wrong:
        reasons.append("LEXICAL_DOMINANCE")   # Lexical wins, both semantic fail

    # Secondary: what did the model incorrectly predict?
    top1_neg_type = ft_result.get("top1_neg_type", "")
    if top1_neg_type.startswith("spec_diff"):
        reasons.append("HARD_NEG_CONFUSION")
    elif top1_neg_type == "synonym":
        reasons.append("SYNONYM_CONFUSION")
    elif top1_neg_type == "abbreviation":
        reasons.append("ABBREVIATION_CONFUSION")
    elif top1_neg_type == "lang_mix":
        reasons.append("LANG_MIX_CONFUSION")
    elif top1_neg_type == "vendor_sku":
        reasons.append("SKU_CONFUSION")

    # Score margin — if very small, it's a close call
    margin = ft_result.get("margin")
    if margin is not None and margin < 0.05:
        reasons.append("LOW_MARGIN_AMBIGUOUS")

    return "; ".join(reasons) if reasons else "UNCLASSIFIED", {
        "top1_neg_type": top1_neg_type,
        "margin": margin,
        "correct_score": ft_result.get("correct_score"),
        "top1_score": ft_result.get("top1_score"),
    }


def main():
    print("Synthetic Failure Analysis")
    print("=" * 60)

    # Load test_synthetic queries
    with open("data/test_synthetic.json", encoding="utf-8") as f:
        queries = json.load(f)

    matched_queries = [q for q in queries if q.get("po_line_id") is not None]
    print(f"  {len(matched_queries)} matched queries in test_synthetic")

    # Encode with all three models
    print("\nEncoding with lexical baseline...")
    lex_results = lexical_rank(matched_queries)

    print("Loading pretrained model and encoding...")
    model_pre = SentenceTransformer(MODEL_PRE)
    pre_results = encode_and_rank(model_pre, matched_queries)

    print("Loading fine-tuned model and encoding...")
    model_ft = SentenceTransformer(str(MODEL_FT))
    ft_results = encode_and_rank(model_ft, matched_queries)

    # Classify failures
    failure_cases = []
    per_diff = defaultdict(lambda: {"total": 0, "lex_correct": 0, "pre_correct": 0, "ft_correct": 0})

    for q in matched_queries:
        qid  = q["query_id"]
        diff = q["difficulty"]
        per_diff[diff]["total"] += 1

        lex = lex_results.get(qid, {})
        pre = pre_results.get(qid, {})
        ft  = ft_results.get(qid, {})

        if lex.get("top1_correct"): per_diff[diff]["lex_correct"] += 1
        if pre.get("top1_correct"): per_diff[diff]["pre_correct"] += 1
        if ft.get("top1_correct"):  per_diff[diff]["ft_correct"]  += 1

        if not ft.get("top1_correct", True):
            classification, detail = classify_failure(q, lex, pre, ft)
            failure_cases.append({
                "query_id":      qid,
                "difficulty":    diff,
                "invoice_line":  q["invoice_line"],
                "correct_po":    q["po_line_id"],
                "correct_desc":  next((c["description"] for c in q["candidates"] if c["is_correct"]), None),
                "predicted_po":  ft.get("top1"),
                "predicted_desc":next((c["description"] for c in q["candidates"]
                                       if c["po_line_id"] == ft.get("top1")), None),
                "top1_neg_type": ft.get("top1_neg_type"),
                "top1_score":    ft.get("top1_score"),
                "correct_score": ft.get("correct_score"),
                "margin":        ft.get("margin"),
                "rank_of_correct": ft.get("rank_of_correct"),
                "lexical_correct":   lex.get("top1_correct"),
                "pretrained_correct": pre.get("top1_correct"),
                "classification": classification,
                "detail": detail,
            })

    # Print per-difficulty breakdown
    print("\nPer-difficulty breakdown (lexical | pretrained | fine-tuned):")
    print(f"  {'Diff':<14} {'N':>4}  {'Lex':>6}  {'Pre':>6}  {'FT':>6}  {'FT-Pre':>8}")
    for diff in ["easy", "medium", "hard", "adversarial"]:
        d = per_diff.get(diff, {})
        n  = d.get("total", 0)
        if n == 0: continue
        l  = d.get("lex_correct", 0) / n
        p  = d.get("pre_correct", 0) / n
        ft = d.get("ft_correct",  0) / n
        delta = ft - p
        print(f"  {diff:<14} {n:>4}  {l:.3f}  {p:.3f}  {ft:.3f}  {delta:+.3f}")

    # Print failure summary
    print(f"\n{len(failure_cases)} failures out of {len(matched_queries)} matched queries")

    class_counts = Counter(c["classification"] for c in failure_cases)
    print("\nFailure classification distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    # Print detailed failure cases
    print("\nDetailed failure cases:")
    for fc in failure_cases:
        print(f"\n  [{fc['query_id']}] {fc['difficulty'].upper()} — {fc['classification']}")
        print(f"    Invoice:   {fc['invoice_line']}")
        print(f"    Correct:   {fc['correct_desc']}")
        print(f"    Predicted: {fc['predicted_desc']}")
        print(f"    Scores:    correct={fc['correct_score']}, predicted={fc['top1_score']}, margin={fc['margin']}")
        print(f"    neg_type:  {fc['top1_neg_type']}")

    # Save
    output = {
        "total_queries": len(matched_queries),
        "total_failures": len(failure_cases),
        "top1_accuracy": round(1 - len(failure_cases)/len(matched_queries), 4),
        "per_difficulty": {diff: {
            "n": d["total"],
            "lex_acc":  round(d["lex_correct"]/d["total"], 4) if d["total"] else None,
            "pre_acc":  round(d["pre_correct"]/d["total"], 4) if d["total"] else None,
            "ft_acc":   round(d["ft_correct"] /d["total"], 4) if d["total"] else None,
            "ft_vs_pre":round((d["ft_correct"]-d["pre_correct"])/d["total"],4) if d["total"] else None,
        } for diff, d in per_diff.items()},
        "classification_counts": dict(class_counts),
        "failure_cases": failure_cases,
    }
    with open(RESULTS_DIR / "synthetic_failure_cases.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: results/synthetic_failure_cases.json")


if __name__ == "__main__":
    main()
