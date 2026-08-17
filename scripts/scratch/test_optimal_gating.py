"""
Analyze optimal gating condition:
Option 1: Hard routing on zero-overlap / tie (tau_m=0.0, tau_s=0.0)
Option 2: Blended scoring on zero-overlap / low-margin
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

val = json.load(open('data/val.json', encoding='utf-8'))
syn = json.load(open('data/test_synthetic_v2.json', encoding='utf-8'))
hrd = json.load(open('data/test_hard_neg.json', encoding='utf-8'))
adv = json.load(open('data/test_adversarial_v2.json', encoding='utf-8'))
hum = json.load(open('data/test_human_v1.json', encoding='utf-8'))

datasets = {'val': val, 'syn': syn, 'hrd': hrd, 'adv': adv, 'hum': hum}
model = SentenceTransformer('models/finetuned_v2_seed44')

def norm(t):
    t = t.lower(); t = re.sub(r'[^\w\s]', ' ', t); return re.sub(r'\s+', ' ', t).strip()
def jacc(a, b):
    ta = set(norm(a).split()); tb = set(norm(b).split())
    return len(ta&tb)/len(ta|tb) if (ta and tb) else 0.0

all_texts = list(set([q['invoice_line'] for qs in datasets.values() for q in qs] + [c['description'] for qs in datasets.values() for q in qs for c in q['candidates']]))
embs = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
t2e = {t: embs[i] for i, t in enumerate(all_texts)}

print("Evaluating Gating Policies:")
for policy_name, tm, ts, alpha in [
    ("Pure Zero/Tie Routing (tau_m=0.0, tau_s=0.0)", 0.0, 0.0, 1.0),
    ("Low Margin Routing (tau_m=0.05, tau_s=0.10)", 0.05, 0.10, 1.0),
    ("Moderate Routing (tau_m=0.10, tau_s=0.15)", 0.10, 0.15, 1.0),
    ("Blended Low Margin (tau_m=0.05, tau_s=0.10, alpha=0.5)", 0.05, 0.10, 0.5),
    ("Blended Moderate (tau_m=0.10, tau_s=0.15, alpha=0.5)", 0.10, 0.15, 0.5),
    ("Blended Soft (tau_m=0.10, tau_s=0.20, alpha=0.3)", 0.10, 0.20, 0.3),
]:
    print(f"\n--- Policy: {policy_name} ---")
    for dname, qs in datasets.items():
        matched = [q for q in qs if q.get('po_line_id')]
        c_hybrid = 0
        routed = 0
        for q in matched:
            inv = q['invoice_line']; cid = q['po_line_id']
            cands = q['candidates']
            l_scores = {c['po_line_id']: jacc(inv, c['description']) for c in cands}
            l_ranked = sorted([(l_scores[c['po_line_id']], c['po_line_id']) for c in cands], key=lambda x: (-x[0], x[1]))
            l_top1 = l_ranked[0][0]
            l_top2 = l_ranked[1][0] if len(l_ranked) > 1 else 0.0
            l_margin = l_top1 - l_top2
            
            is_ambiguous = (l_margin <= tm) or (l_top1 <= ts)
            if is_ambiguous:
                routed += 1
                scored = []
                for c in cands:
                    pid = c['po_line_id']
                    sem_s = float(np.dot(t2e[inv], t2e[c['description']]))
                    lex_s = l_scores[pid]
                    final_s = alpha * sem_s + (1.0 - alpha) * lex_s
                    scored.append((final_s, pid))
                scored_ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
                winner = scored_ranked[0][1]
            else:
                winner = l_ranked[0][1]
            
            if winner == cid:
                c_hybrid += 1
        print(f"  {dname:5}: Top-1={c_hybrid}/{len(matched)} ({c_hybrid/len(matched):.4f}) | Routed={routed}/{len(matched)} ({routed/len(matched)*100:.1f}%)")
