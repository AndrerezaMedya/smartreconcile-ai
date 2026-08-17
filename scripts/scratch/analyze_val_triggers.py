"""
Analyze validation set to explore conditional gating triggers.
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

# Load best FT model (seed 44)
model = SentenceTransformer("models/finetuned_v2_seed44")

print(f"Validation queries: {len(val_data)}")

# Collect all descriptions for encoding
all_texts = set()
for q in val_data:
    all_texts.add(q["invoice_line"])
    for c in q["candidates"]:
        all_texts.add(c["description"])

text_list = list(all_texts)
embs = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
text2emb = {t: embs[i] for i, t in enumerate(text_list)}

val_rows = []
for q in val_data:
    if q.get("po_line_id") is None:
        continue
    inv = q["invoice_line"]
    correct_id = q["po_line_id"]
    inv_emb = text2emb[inv]
    
    # Lexical scoring
    lex_scored = [(jaccard(inv, c["description"]), c["po_line_id"], c["description"]) for c in q["candidates"]]
    lex_ranked = sorted(lex_scored, key=lambda x: (-x[0], x[1]))
    
    lex_top1_score = lex_ranked[0][0]
    lex_top2_score = lex_ranked[1][0] if len(lex_ranked) > 1 else 0.0
    lex_margin = lex_top1_score - lex_top2_score
    lex_correct = (lex_ranked[0][1] == correct_id)
    
    # Semantic scoring
    sem_scored = [(float(np.dot(inv_emb, text2emb[c["description"]])), c["po_line_id"], c["description"]) for c in q["candidates"]]
    sem_ranked = sorted(sem_scored, key=lambda x: (-x[0], x[1]))
    
    sem_top1_score = sem_ranked[0][0]
    sem_top2_score = sem_ranked[1][0] if len(sem_ranked) > 1 else 0.0
    sem_margin = sem_top1_score - sem_top2_score
    sem_correct = (sem_ranked[0][1] == correct_id)
    
    val_rows.append({
        "query_id": q["query_id"],
        "invoice": inv,
        "difficulty": q.get("difficulty"),
        "lex_top1": round(lex_top1_score, 4),
        "lex_top2": round(lex_top2_score, 4),
        "lex_margin": round(lex_margin, 4),
        "lex_correct": lex_correct,
        "sem_top1": round(sem_top1_score, 4),
        "sem_top2": round(sem_top2_score, 4),
        "sem_margin": round(sem_margin, 4),
        "sem_correct": sem_correct,
        "lex_winner": lex_ranked[0][2],
        "sem_winner": sem_ranked[0][2],
        "correct_desc": next(c["description"] for c in q["candidates"] if c["po_line_id"] == correct_id)
    })

print(f"\nLexical accuracy on val: {sum(1 for r in val_rows if r['lex_correct'])}/{len(val_rows)} = {sum(1 for r in val_rows if r['lex_correct'])/len(val_rows):.4f}")
print(f"Semantic accuracy on val: {sum(1 for r in val_rows if r['sem_correct'])}/{len(val_rows)} = {sum(1 for r in val_rows if r['sem_correct'])/len(val_rows):.4f}")

print("\nQueries where Lexical was WRONG:")
for r in val_rows:
    if not r["lex_correct"]:
        print(f"  [{r['query_id']}] ({r['difficulty']}) inv='{r['invoice']}' | lex_margin={r['lex_margin']} lex_top1={r['lex_top1']}")
        print(f"     Lexical picked: {r['lex_winner']}")
        print(f"     Semantic picked: {r['sem_winner']} (sem_correct={r['sem_correct']})")
        print(f"     Target PO: {r['correct_desc']}")

print("\nQueries where Semantic was WRONG but Lexical was CORRECT:")
for r in val_rows:
    if r["lex_correct"] and not r["sem_correct"]:
        print(f"  [{r['query_id']}] ({r['difficulty']}) inv='{r['invoice']}' | lex_margin={r['lex_margin']} lex_top1={r['lex_top1']}")
        print(f"     Lexical picked: {r['lex_winner']}")
        print(f"     Semantic picked: {r['sem_winner']}")
        print(f"     Target PO: {r['correct_desc']}")
