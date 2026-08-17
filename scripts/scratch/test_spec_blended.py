"""
Test Spec-Aware Blended Routing inside the ambiguous subset.
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

SPEC_PATTERN = re.compile(r"\b(\d+[\w./\-]*|\w+\d+\w*)\b")

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

def spec_overlap_score(a: str, b: str) -> float:
    ta = set(SPEC_PATTERN.findall(normalize(a)))
    tb = set(SPEC_PATTERN.findall(normalize(b)))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)

# Load data
with open("data/val.json", encoding="utf-8") as f:
    val_data = json.load(f)

with open("data/test_synthetic_v2.json", encoding="utf-8") as f:
    syn_data = json.load(f)

with open("data/test_hard_neg.json", encoding="utf-8") as f:
    hard_data = json.load(f)

with open("data/test_adversarial_v2.json", encoding="utf-8") as f:
    adv_data = json.load(f)

with open("data/test_human_v1.json", encoding="utf-8") as f:
    hum_data = json.load(f)

model = SentenceTransformer("models/finetuned_v2_seed44")

datasets = {
    "val": val_data,
    "test_synthetic_v2": syn_data,
    "test_hard_neg": hard_data,
    "test_adversarial_v2": adv_data,
    "test_human_v1": hum_data,
}

all_texts = set()
for dname, qs in datasets.items():
    for q in qs:
        all_texts.add(q["invoice_line"])
        for c in q.get("candidates", []):
            all_texts.add(c["description"])

text_list = list(all_texts)
embs = model.encode(text_list, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
text2emb = {t: embs[i] for i, t in enumerate(text_list)}

def eval_config(tau_m, tau_s, alpha, beta):
    out = {}
    for dname, qs in datasets.items():
        matched = [q for q in qs if q.get("po_line_id")]
        correct = 0
        routed = 0
        for q in matched:
            inv = q["invoice_line"]
            cid_true = q["po_line_id"]
            inv_emb = text2emb[inv]
            cands = q["candidates"]
            
            lex_scores = {c["po_line_id"]: jaccard(inv, c["description"]) for c in cands}
            lex_ranked = sorted([(lex_scores[c["po_line_id"]], c["po_line_id"]) for c in cands], key=lambda x: (-x[0], x[1]))
            lex_top1 = lex_ranked[0][0]
            lex_top2 = lex_ranked[1][0] if len(lex_ranked) > 1 else 0.0
            lex_margin = lex_top1 - lex_top2
            
            is_ambiguous = (lex_margin <= tau_m) or (lex_top1 <= tau_s)
            if is_ambiguous:
                routed += 1
                scored = []
                for c in cands:
                    pid = c["po_line_id"]
                    sem_s = float(np.dot(inv_emb, text2emb[c["description"]]))
                    lex_s = lex_scores[pid]
                    spec_s = spec_overlap_score(inv, c["description"])
                    final_s = alpha * sem_s + (1.0 - alpha) * lex_s + beta * spec_s
                    scored.append((final_s, pid))
                scored_ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
                winner = scored_ranked[0][1]
            else:
                winner = lex_ranked[0][1]
            
            if winner == cid_true:
                correct += 1
        out[dname] = {
            "top1": correct / len(matched),
            "routed_pct": (routed / len(matched)) * 100
        }
    return out

# Sweep alpha & beta for ambiguity resolution
print(f"{'tau_m':>6} {'tau_s':>6} {'alpha':>6} {'beta':>6} | {'val':>7} {'syn_v2':>7} {'hard':>7} {'adv_v2':>7} {'human':>7}")
print("-" * 75)

for tm in [0.0, 0.05, 0.10]:
    for ts in [0.0, 0.10, 0.15, 0.20]:
        for alpha in [0.3, 0.5, 0.7, 1.0]:
            for beta in [0.0, 0.10, 0.20]:
                res = eval_config(tm, ts, alpha, beta)
                val_t1 = res["val"]["top1"]
                syn_t1 = res["test_synthetic_v2"]["top1"]
                hrd_t1 = res["test_hard_neg"]["top1"]
                adv_t1 = res["test_adversarial_v2"]["top1"]
                hum_t1 = res["test_human_v1"]["top1"]
                if val_t1 >= 0.85 and syn_t1 >= 0.90:
                    print(f"{tm:>6.2f} {ts:>6.2f} {alpha:>6.2f} {beta:>6.2f} | {val_t1:>7.4f} {syn_t1:>7.4f} {hrd_t1:>7.4f} {adv_t1:>7.4f} {hum_t1:>7.4f}")
