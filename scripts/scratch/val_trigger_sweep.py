"""
Trigger parameter sweep and calibration on validation data ONLY.
Evaluates various gating strategies on val.json:
  - Trigger condition: (lex_top1 < tau_top1) or (lex_margin < tau_margin)
  - Hard routing vs Blended reranking inside ambiguous subset
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

# Load val.json
with open("data/val.json", encoding="utf-8") as f:
    val_data = json.load(f)

# Load models to compare (seed 44, seed 43, seed 42)
models = {
    "ft_v2_s44": SentenceTransformer("models/finetuned_v2_seed44"),
    "ft_v2_s43": SentenceTransformer("models/finetuned_v2_seed43"),
}

# Pre-encode val texts
all_texts = set()
for q in val_data:
    all_texts.add(q["invoice_line"])
    for c in q["candidates"]:
        all_texts.add(c["description"])

text_list = list(all_texts)

for m_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Trigger Sweep on VAL for model: {m_name}")
    print(f"{'='*60}")
    
    embs = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    text2emb = {t: embs[i] for i, t in enumerate(text_list)}
    
    matched_queries = [q for q in val_data if q.get("po_line_id")]
    N = len(matched_queries)
    
    # Precompute lexical and semantic scores
    val_items = []
    for q in matched_queries:
        inv = q["invoice_line"]
        correct_id = q["po_line_id"]
        inv_emb = text2emb[inv]
        
        cands = q["candidates"]
        lex_scores = {c["po_line_id"]: jaccard(inv, c["description"]) for c in cands}
        sem_scores = {c["po_line_id"]: float(np.dot(inv_emb, text2emb[c["description"]])) for c in cands}
        
        # Sort lexical
        lex_ranked = sorted([(lex_scores[c["po_line_id"]], c["po_line_id"]) for c in cands], key=lambda x: (-x[0], x[1]))
        lex_top1 = lex_ranked[0][0]
        lex_top2 = lex_ranked[1][0] if len(lex_ranked) > 1 else 0.0
        lex_margin = lex_top1 - lex_top2
        
        # Sort semantic
        sem_ranked = sorted([(sem_scores[c["po_line_id"]], c["po_line_id"]) for c in cands], key=lambda x: (-x[0], x[1]))
        sem_top1 = sem_ranked[0][0]
        sem_top2 = sem_ranked[1][0] if len(sem_ranked) > 1 else 0.0
        sem_margin = sem_top1 - sem_top2
        
        val_items.append({
            "qid": q["query_id"],
            "correct_id": correct_id,
            "cands": cands,
            "lex_scores": lex_scores,
            "sem_scores": sem_scores,
            "lex_top1": lex_top1,
            "lex_margin": lex_margin,
            "lex_winner": lex_ranked[0][1],
            "sem_winner": sem_ranked[0][1],
            "lex_correct": (lex_ranked[0][1] == correct_id),
            "sem_correct": (sem_ranked[0][1] == correct_id),
        })
    
    print(f"Total matched queries on val: {N}")
    print(f"Lexical pure top1: {sum(1 for it in val_items if it['lex_correct'])}/{N} ({sum(1 for it in val_items if it['lex_correct'])/N:.4f})")
    print(f"Semantic pure top1: {sum(1 for it in val_items if it['sem_correct'])}/{N} ({sum(1 for it in val_items if it['sem_correct'])/N:.4f})")
    
    # Grid of triggers:
    # Condition: route to semantic IF (lex_margin <= tau_m) OR (lex_top1 <= tau_s)
    tau_margins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    tau_top1s = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35]
    
    results = []
    for tm in tau_margins:
        for ts in tau_top1s:
            # Evaluate Hard routing
            correct_hard = 0
            routed_count = 0
            changed_count = 0
            improved_count = 0
            degraded_count = 0
            
            for it in val_items:
                is_ambiguous = (it["lex_margin"] <= tm) or (it["lex_top1"] <= ts)
                if is_ambiguous:
                    routed_count += 1
                    final_winner = it["sem_winner"]
                else:
                    final_winner = it["lex_winner"]
                
                is_correct = (final_winner == it["correct_id"])
                if is_correct:
                    correct_hard += 1
                
                if final_winner != it["lex_winner"]:
                    changed_count += 1
                    if is_correct and not it["lex_correct"]:
                        improved_count += 1
                    elif not is_correct and it["lex_correct"]:
                        degraded_count += 1
            
            top1_acc = correct_hard / N
            routed_pct = (routed_count / N) * 100
            
            results.append({
                "tau_margin": tm,
                "tau_top1": ts,
                "top1_acc": top1_acc,
                "routed_pct": routed_pct,
                "changed": changed_count,
                "improved": improved_count,
                "degraded": degraded_count,
            })
    
    # Sort by top1_acc descending, then routed_pct descending (preferring more active semantic usage when tied)
    results.sort(key=lambda x: (-x["top1_acc"], -x["routed_pct"], x["degraded"]))
    
    print("\nTop 10 Trigger Configurations on Val (Hard Routing):")
    print(f"  {'tau_m':>6} {'tau_top1':>8} {'val_top1':>10} {'routed%':>10} {'changed':>8} {'improved':>9} {'degraded':>9}")
    print("  " + "-"*65)
    for r in results[:10]:
        print(f"  {r['tau_margin']:>6.2f} {r['tau_top1']:>8.2f} {r['top1_acc']:>10.4f} {r['routed_pct']:>9.1f}% {r['changed']:>8} {r['improved']:>9} {r['degraded']:>9}")
