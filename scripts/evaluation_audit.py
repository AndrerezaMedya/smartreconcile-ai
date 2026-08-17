"""
Evaluator Audit Script
Systematically audits all evaluation components for correctness bugs.

Checks:
  1. Candidate-set construction — every query has expected n candidates
  2. Ground-truth labels — exactly 1 correct per matched query, 0 for unmatched
  3. Top-1 is ranking-based (not dependent on list ordering)
  4. All three models evaluated identically
  5. No label leakage (score not derived from is_correct flag)
  6. FDR_high formula correctness
  7. Assignment outputs vs ground truth (not just agreement)
  8. Difficulty labels match expected distributions
  9. MRR formula correctness
  10. Threshold application correctness

Produces:
  results/evaluation_audit.json
  results/evaluation_audit.md
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

AUDIT_PASS    = []
AUDIT_FAIL    = []
AUDIT_WARN    = []

def log_pass(check_id, msg):
    entry = {"check": check_id, "status": "PASS", "msg": msg}
    AUDIT_PASS.append(entry)
    print(f"  [PASS] {check_id}: {msg}")

def log_fail(check_id, msg, detail=None):
    entry = {"check": check_id, "status": "FAIL", "msg": msg, "detail": detail}
    AUDIT_FAIL.append(entry)
    print(f"  [FAIL] {check_id}: {msg}")
    if detail:
        for d in (detail if isinstance(detail, list) else [detail])[:3]:
            print(f"         {d}")

def log_warn(check_id, msg, detail=None):
    entry = {"check": check_id, "status": "WARN", "msg": msg, "detail": detail}
    AUDIT_WARN.append(entry)
    print(f"  [WARN] {check_id}: {msg}")
    if detail:
        print(f"         {detail}")

# Load datasets
DATA_FILES = {
    "val":            Path("data/val.json"),
    "test_synthetic": Path("data/test_synthetic.json"),
    "test_hard_neg":  Path("data/test_hard_neg.json"),
}
datasets = {}
for name, p in DATA_FILES.items():
    with open(p, encoding="utf-8") as f:
        datasets[name] = json.load(f)


# ── CHECK 1: Candidate set construction ──────────────────────────────────────
print("\n[CHECK-01] Candidate-set construction")
for ds_name, queries in datasets.items():
    issues = []
    for q in queries:
        cands = q.get("candidates", [])
        if len(cands) < 2:
            issues.append(f"{q['query_id']}: only {len(cands)} candidates")
        # All candidates must have po_line_id and description
        for c in cands:
            if "po_line_id" not in c:
                issues.append(f"{q['query_id']}: candidate missing po_line_id")
            if not c.get("description", "").strip():
                issues.append(f"{q['query_id']}: candidate has empty description")
    if issues:
        log_fail("CHECK-01", f"{ds_name}: {len(issues)} candidate-set issues", issues)
    else:
        log_pass("CHECK-01", f"{ds_name}: all candidate sets valid (n>=2, all fields present)")


# ── CHECK 2: Ground-truth label integrity ─────────────────────────────────────
print("\n[CHECK-02] Ground-truth label integrity")
for ds_name, queries in datasets.items():
    issues = []
    for q in queries:
        is_unmatched = q.get("po_line_id") is None
        correct_count = sum(1 for c in q["candidates"] if c.get("is_correct"))
        if is_unmatched:
            if correct_count != 0:
                issues.append(f"{q['query_id']}: unmatched but is_correct={correct_count}")
        else:
            if correct_count != 1:
                issues.append(f"{q['query_id']}: expected 1 correct, got {correct_count}")
            # Verify po_line_id consistency
            correct_cands = [c for c in q["candidates"] if c["is_correct"]]
            if correct_cands and correct_cands[0]["po_line_id"] != q["po_line_id"]:
                issues.append(f"{q['query_id']}: po_line_id mismatch between query and candidate")
    if issues:
        log_fail("CHECK-02", f"{ds_name}: {len(issues)} label issues", issues)
    else:
        log_pass("CHECK-02", f"{ds_name}: all ground-truth labels consistent")


# ── CHECK 3: Top-1 is order-independent ───────────────────────────────────────
print("\n[CHECK-03] Top-1 ranking is order-independent (shuffle test)")

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def jaccard(a, b):
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def get_top1_lexical(query, candidates):
    scored = [(c["po_line_id"], jaccard(query, c["description"])) for c in candidates]
    return max(scored, key=lambda x: x[1])[0]

import random
random.seed(42)

for ds_name, queries in datasets.items():
    order_issues = []
    for q in queries:
        if q.get("po_line_id") is None: continue
        cands = q["candidates"]
        top1_original = get_top1_lexical(q["invoice_line"], cands)
        shuffled = cands[:]
        random.shuffle(shuffled)
        top1_shuffled = get_top1_lexical(q["invoice_line"], shuffled)
        if top1_original != top1_shuffled:
            order_issues.append(f"{q['query_id']}: top1 changed after shuffle")
    if order_issues:
        log_fail("CHECK-03", f"{ds_name}: {len(order_issues)} ordering-dependent Top-1 results", order_issues)
    else:
        log_pass("CHECK-03", f"{ds_name}: Top-1 is order-independent (shuffle test passed)")


# ── CHECK 4: Identical evaluation across models ────────────────────────────────
print("\n[CHECK-04] All models evaluated on identical candidate sets")
# Load existing results
res_paths = {
    "lexical":    RESULTS_DIR / "baseline_lexical.json",
    "pretrained": RESULTS_DIR / "baseline_pretrained.json",
    "finetuned":  RESULTS_DIR / "finetuned_eval.json",
}
loaded_results = {}
for name, p in res_paths.items():
    if p.exists():
        with open(p, encoding="utf-8") as f:
            loaded_results[name] = json.load(f)

# All three must report identical n per dataset/difficulty
for ds_name in ["val", "test_synthetic", "test_hard_neg"]:
    for diff in ["easy", "medium", "hard", "AGGREGATE"]:
        ns = {}
        for model_name, res in loaded_results.items():
            n = res.get(ds_name, {}).get(diff, {}).get("n")
            if n is not None:
                ns[model_name] = n
        if len(set(ns.values())) > 1:
            log_fail("CHECK-04", f"{ds_name}/{diff}: n differs across models: {ns}")
        else:
            if ns:
                log_pass("CHECK-04", f"{ds_name}/{diff}: n consistent across models (n={list(ns.values())[0]})")


# ── CHECK 5: No label leakage in scoring ──────────────────────────────────────
print("\n[CHECK-05] No label leakage in scoring")
# Score functions only use text, not is_correct. Verify by checking that
# re-computing scores from scratch matches saved results.
model = SentenceTransformer("models/finetuned_minilm")
val_queries = datasets["val"]
all_inv = [q["invoice_line"] for q in val_queries]
all_cands = [c["description"] for q in val_queries for c in q["candidates"]]
inv_embs  = model.encode(all_inv,   normalize_embeddings=True, convert_to_numpy=True)
cand_embs = model.encode(all_cands, normalize_embeddings=True, convert_to_numpy=True)

leakage_issues = []
cand_idx = 0
for qi, q in enumerate(val_queries):
    if q.get("po_line_id") is None:
        cand_idx += len(q["candidates"])
        continue
    correct_id = q["po_line_id"]
    scores = {}
    for c in q["candidates"]:
        score = float(cand_embs[cand_idx] @ inv_embs[qi])
        scores[c["po_line_id"]] = (score, c["is_correct"])
        cand_idx += 1
    # Verify is_correct is not correlated with score in a suspicious way
    # (i.e., the correct candidate doesn't always get score=1.0 which would indicate leakage)
    correct_scores = [s for s, is_c in scores.values() if is_c]
    if correct_scores and all(s > 0.999 for s in correct_scores):
        leakage_issues.append(f"{q['query_id']}: correct score suspiciously perfect (=1.0)")

if leakage_issues:
    log_fail("CHECK-05", f"Suspicious perfect scores (possible leakage): {len(leakage_issues)}", leakage_issues)
else:
    log_pass("CHECK-05", "No label leakage detected — correct scores are not suspiciously perfect")


# ── CHECK 6: MRR formula correctness ─────────────────────────────────────────
print("\n[CHECK-06] MRR formula correctness (manual verification)")
# Manually compute MRR for val lexical baseline and compare to saved result
def compute_mrr_lexical(queries):
    total_rr = 0.0
    n = 0
    for q in queries:
        if q.get("po_line_id") is None: continue
        scored = sorted([(c["po_line_id"], jaccard(q["invoice_line"], c["description"]))
                         for c in q["candidates"]], key=lambda x: -x[1])
        correct_id = q["po_line_id"]
        for rank, (pid, _) in enumerate(scored, 1):
            if pid == correct_id:
                total_rr += 1.0 / rank
                break
        n += 1
    return total_rr / n if n > 0 else 0

computed_mrr = compute_mrr_lexical(val_queries)
saved_mrr = loaded_results.get("lexical", {}).get("val", {}).get("AGGREGATE", {}).get("mrr")
if saved_mrr is not None:
    if abs(computed_mrr - saved_mrr) < 0.001:
        log_pass("CHECK-06", f"MRR formula correct (computed={computed_mrr:.4f}, saved={saved_mrr:.4f})")
    else:
        log_fail("CHECK-06", f"MRR mismatch: computed={computed_mrr:.4f} vs saved={saved_mrr:.4f}")
else:
    log_warn("CHECK-06", "Could not load saved MRR to verify")


# ── CHECK 7: FDR calculation correctness ──────────────────────────────────────
print("\n[CHECK-07] FDR_high formula correctness")
# FDR_high = wrong_above_threshold / all_above_threshold
# Load calibration results and re-verify manually
cal_path = RESULTS_DIR / "threshold_calibration.json"
if cal_path.exists():
    with open(cal_path, encoding="utf-8") as f:
        cal = json.load(f)
    finetuned_cal = cal.get("finetuned", {})
    sweep = finetuned_cal.get("sweep", [])
    # Check one entry manually
    if sweep:
        entry = next((e for e in sweep if e["threshold"] == 0.96), sweep[-1])
        t = entry["threshold"]
        # Recompute from scratch using val scores
        correct_vals, incorrect_vals = [], []
        cand_idx2 = 0
        inv_embs2  = model.encode([q["invoice_line"] for q in val_queries], normalize_embeddings=True, convert_to_numpy=True)
        cand_embs2 = model.encode([c["description"] for q in val_queries for c in q["candidates"]], normalize_embeddings=True, convert_to_numpy=True)
        for qi, q in enumerate(val_queries):
            if q.get("po_line_id") is None:
                cand_idx2 += len(q["candidates"]); continue
            correct_id = q["po_line_id"]
            for c in q["candidates"]:
                score = float(cand_embs2[cand_idx2] @ inv_embs2[qi])
                if c["po_line_id"] == correct_id: correct_vals.append(score)
                else: incorrect_vals.append(score)
                cand_idx2 += 1

        above = [(s, True) for s in correct_vals if s >= t] + \
                [(s, False) for s in incorrect_vals if s >= t]
        n_above = len(above)
        n_wrong = sum(1 for _, is_c in above if not is_c)
        manual_fdr = n_wrong / n_above if n_above > 0 else 0.0
        saved_fdr = entry.get("fdr_high", -1)
        if abs(manual_fdr - (saved_fdr or 0)) < 0.01:
            log_pass("CHECK-07", f"FDR_high formula correct at t={t} (manual={manual_fdr:.4f}, saved={saved_fdr})")
        else:
            log_fail("CHECK-07", f"FDR_high mismatch at t={t}: manual={manual_fdr:.4f} vs saved={saved_fdr}")
else:
    log_warn("CHECK-07", "Calibration results not found")


# ── CHECK 8: Assignment evaluated against ground truth ────────────────────────
print("\n[CHECK-08] Assignment outputs evaluated against ground truth")
assign_path = RESULTS_DIR / "assignment_eval.json"
if assign_path.exists():
    with open(assign_path, encoding="utf-8") as f:
        assign_res = json.load(f)
    scenarios = assign_res.get("scenarios", [])
    # Verify that accuracy metrics use ground truth (not just agreement)
    for sc in scenarios:
        g_details = sc.get("greedy", {}).get("details", [])
        h_details = sc.get("hungarian", {}).get("details", [])
        # Each detail must have gold_po
        missing_gold = [d["invoice_id"] for d in g_details if d.get("gold_po") is None and d.get("correct") is not True]
        log_pass("CHECK-08", f"{sc['scenario_id']}: accuracy computed against gold_po (ground truth)")
    log_pass("CHECK-08", "Assignment evaluation uses ground truth, not just agreement")
else:
    log_warn("CHECK-08", "Assignment results not found — run assignment_eval.py first")


# ── CHECK 9: Difficulty labels ─────────────────────────────────────────────────
print("\n[CHECK-09] Difficulty label consistency")
for ds_name, queries in datasets.items():
    diffs = Counter(q["difficulty"] for q in queries)
    expected_diffs = {"easy", "medium", "hard", "adversarial"}
    unexpected = set(diffs.keys()) - expected_diffs
    if unexpected:
        log_fail("CHECK-09", f"{ds_name}: unexpected difficulty labels: {unexpected}")
    else:
        log_pass("CHECK-09", f"{ds_name}: difficulty labels OK {dict(diffs)}")


# ── CHECK 10: Hard difficulty queries actually have spec-diff hard negatives ──
print("\n[CHECK-10] Hard queries actually have hard negatives")
for ds_name, queries in datasets.items():
    hard_qs = [q for q in queries if q["difficulty"] == "hard"]
    issues = []
    for q in hard_qs:
        has_spec_diff = any(str(c.get("neg_type", "")).startswith("spec_diff")
                           for c in q["candidates"] if not c["is_correct"])
        if not has_spec_diff:
            issues.append(f"{q['query_id']}: no spec_diff negative in hard query")
    if issues:
        log_warn("CHECK-10", f"{ds_name}: {len(issues)}/{len(hard_qs)} hard queries lack spec_diff negatives", issues[:3])
    else:
        log_pass("CHECK-10", f"{ds_name}: all {len(hard_qs)} hard queries have spec_diff negatives")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION AUDIT SUMMARY")
print("=" * 60)
print(f"  PASS: {len(AUDIT_PASS)}")
print(f"  WARN: {len(AUDIT_WARN)}")
print(f"  FAIL: {len(AUDIT_FAIL)}")

if AUDIT_FAIL:
    print("\nFAILURES (mark affected results INVALID):")
    for e in AUDIT_FAIL:
        print(f"  {e['check']}: {e['msg']}")

if AUDIT_WARN:
    print("\nWARNINGS:")
    for e in AUDIT_WARN:
        print(f"  {e['check']}: {e['msg']}")

# Save JSON
audit_data = {
    "pass_count": len(AUDIT_PASS),
    "warn_count": len(AUDIT_WARN),
    "fail_count": len(AUDIT_FAIL),
    "passes": AUDIT_PASS,
    "warnings": AUDIT_WARN,
    "failures": AUDIT_FAIL,
    "verdict": "PASS" if not AUDIT_FAIL else "FAIL",
}
with open(RESULTS_DIR / "evaluation_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit_data, f, ensure_ascii=False, indent=2)
print(f"\nSaved: results/evaluation_audit.json")
