"""
Blended routing sweep inside ambiguous subset on validation data ONLY.
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

model = SentenceTransformer("models/finetuned_v2_seed44")

all_texts = set()
for q in val_data:
    all_texts.add(q["invoice_line"])
    for c in q["candidates"]:
        all_texts.add(c["description"])

text_list = list(all_texts)
embs = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
text2emb = {t: embs[i] for i, t in enumerate(text_list)}

matched_queries = [q for q in val_data if q.get("po_line_id")]
N = len(matched_queries)

val_items = []
for q in matched_queries:
    inv = q["invoice_line"]
    correct_id = q["po_line_id"]
    inv_emb = text2emb[inv]
    
    cands = q["candidates"]
    lex_scores = {c["po_line_id"]: jaccard(inv, c["description"]) for c in cands}
    sem_scores = {c["po_line_id"]: float(np.dot(inv_emb, text2emb[c["description"]])) for c in cands}
    
    lex_ranked = sorted([(lex_scores[c["po_line_id"]], c["po_line_id"]) for c in cands], key=lambda x: (-x[0], x[1]))
    lex_top1 = lex_ranked[0][0]
    lex_top2 = lex_ranked[1][0] if len(lex_ranked) > 1 else 0.0
    lex_margin = lex_top1 - lex_top2
    
    val_items.append({
        "qid": q["query_id"],
        "correct_id": correct_id,
        "cands": cands,
        "lex_scores": lex_scores,
        "sem_scores": sem_scores,
        "lex_top1": lex_top1,
        "lex_margin": lex_margin,
        "lex_winner": lex_ranked[0][1],
        "lex_correct": (lex_ranked[0][1] == correct_id),
    })

print(f"Total matched queries on val: {N}")

# Sweep tau_m, tau_s, and alpha for blended reranking
results = []
for tm in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for ts in [0.0, 0.10, 0.20, 0.25]:
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            correct_count = 0
            routed_count = 0
            changed_count = 0
            improved_count = 0
            degraded_count = 0
            
            for it in val_items:
                is_ambiguous = (it["lex_margin"] <= tm) or (it["lex_top1"] <= ts)
                if is_ambiguous:
                    routed_count += 1
                    # Blend inside ambiguous
                    scored = []
                    for c in it["cands"]:
                        cid = c["po_line_id"]
                        s = alpha * it["sem_scores"][cid] + (1 - alpha) * it["lex_scores"][cid]
                        scored.append((s, cid))
                    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
                    final_winner = ranked[0][1]
                else:
                    final_winner = it["lex_winner"]
                
                is_correct = (final_winner == it["correct_id"])
                if is_correct:
                    correct_count += 1
                
                if final_winner != it["lex_winner"]:
                    changed_count += 1
                    if is_correct and not it["lex_correct"]:
                        improved_count += 1
                    elif not is_correct and it["lex_correct"]:
                        degraded_count += 1
            
            top1_acc = correct_count / N
            routed_pct = (routed_count / N) * 100
            
            results.append({
                "tau_m": tm,
                "tau_s": ts,
                "alpha": alpha,
                "top1_acc": top1_acc,
                "routed_pct": routed_pct,
                "changed": changed_count,
                "improved": improved_count,
                "degraded": degraded_count,
            })

results.sort(key=lambda x: (-x["top1_acc"], -x["routed_pct"], x["degraded"]))

print("\nTop 15 Blended Hybrid Configurations on Val:")
print(f"  {'tau_m':>6} {'tau_s':>6} {'alpha':>6} {'val_top1':>10} {'routed%':>10} {'changed':>8} {'improved':>9} {'degraded':>9}")
print("  " + "-"*75)
for r in results[:15]:
    print(f"  {r['tau_m']:>6.2f} {r['tau_s']:>6.2f} {r['alpha']:>6.2f} {r['top1_acc']:>10.4f} {r['routed_pct']:>9.1f}% {r['changed']:>8} {r['improved']:>9} {r['degraded']:>9}")
