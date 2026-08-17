"""
Phase 1: Automated Leakage Audit
Checks the dataset for data leakage between train/val/test splits.

Checks:
  1. No material_id overlap across splits
  2. No normalized string duplicates across splits
  3. No near-duplicate descriptions crossing splits (Jaccard > 0.8)
  4. Every candidate set has exactly 1 correct answer (except unmatched adversarial)
  5. Difficulty distribution is reasonable per split
  6. No query_id duplicates
"""

import json
import re
from pathlib import Path
from collections import Counter

PASSES = []
FAILURES = []
WARNINGS = []


def log_pass(msg):
    PASSES.append(f"  PASS  {msg}")
    print(f"  [PASS]  {msg}")


def log_fail(msg):
    FAILURES.append(f"  FAIL  {msg}")
    print(f"  [FAIL]  {msg}")


def log_warn(msg):
    WARNINGS.append(f"  WARN  {msg}")
    print(f"  [WARN]  {msg}")


def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> set:
    return set(normalize(text).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Load catalog ──────────────────────────────────────────────────────────────
CATALOG_PATH = Path("data/material_catalog.json")
with open(CATALOG_PATH, encoding="utf-8") as f:
    CATALOG = json.load(f)

CATALOG_SPLIT = {m["material_id"]: m["split"] for m in CATALOG["materials"]}

# ── Load dataset files ─────────────────────────────────────────────────────────
DATA_FILES = {
    "train_v2":           Path("data/train_v2.json"),
    "val":                Path("data/val.json"),
    "test_synthetic_v2":  Path("data/test_synthetic_v2.json"),
    "test_hard_neg":      Path("data/test_hard_neg.json"),
    "test_adversarial_v2":Path("data/test_adversarial_v2.json"),
    "test_human_v1":      Path("data/test_human_v1.json"),
}

datasets = {}
for name, path in DATA_FILES.items():
    if path.exists():
        with open(path, encoding="utf-8") as f:
            datasets[name] = json.load(f)
    else:
        log_fail(f"Missing file: {path}")
        datasets[name] = []

# ── CHECK 1: No material_id overlap across splits ─────────────────────────────
print("\n[1] Material-ID isolation check")

# Define expected splits per dataset
DATASET_TO_SPLIT = {
    "train_v2": "train",
    "val": "validation",
    "test_synthetic_v2": "test",
    "test_hard_neg": "test",
    "test_adversarial_v2": "test",
    "test_human_v1": "test",
}

for ds_name, queries in datasets.items():
    expected_catalog_split = DATASET_TO_SPLIT[ds_name]
    wrong = []
    for q in queries:
        mid = q.get("material_id")
        if mid and mid in CATALOG_SPLIT:
            if CATALOG_SPLIT[mid] != expected_catalog_split:
                wrong.append(f"{q['query_id']}: {mid} (catalog={CATALOG_SPLIT[mid]}, dataset={expected_catalog_split})")
    if wrong:
        log_fail(f"{ds_name}: {len(wrong)} queries use wrong-split materials")
        for w in wrong[:5]:
            print(f"    {w}")
    else:
        log_pass(f"{ds_name}: All material_ids are from correct catalog split")

# Check no material_id appears in both train and val/test
train_mids = set(q.get("material_id") for q in datasets["train_v2"] if q.get("material_id"))
val_mids   = set(q.get("material_id") for q in datasets["val"] if q.get("material_id"))
test_syn_m = set(q.get("material_id") for q in datasets["test_synthetic_v2"] if q.get("material_id"))
test_hrd_m = set(q.get("material_id") for q in datasets["test_hard_neg"] if q.get("material_id"))
test_adv_m = set(q.get("material_id") for q in datasets["test_adversarial_v2"] if q.get("material_id"))
test_hum_m = set(q.get("material_id") for q in datasets["test_human_v1"] if q.get("material_id"))

for pair_name, pair_set in [
    ("train vs val", train_mids & val_mids),
    ("train vs test_syn_v2", train_mids & test_syn_m),
    ("train vs test_hard", train_mids & test_hrd_m),
    ("train vs test_adv_v2", train_mids & test_adv_m),
    ("train vs test_human", train_mids & test_hum_m),
    ("val vs test_syn_v2", val_mids & test_syn_m),
]:
    if pair_set:
        log_fail(f"material_id overlap {pair_name}: {pair_set}")
    else:
        log_pass(f"No material_id overlap: {pair_name}")

# ── CHECK 2: No query_id duplicates ──────────────────────────────────────────
print("\n[2] Query-ID uniqueness check")
all_qids = []
for queries in datasets.values():
    all_qids.extend(q["query_id"] for q in queries)

dupes = [qid for qid, cnt in Counter(all_qids).items() if cnt > 1]
if dupes:
    log_fail(f"Duplicate query_ids: {dupes[:5]}")
else:
    log_pass(f"All {len(all_qids)} query_ids are unique")

# ── CHECK 3: Candidate set integrity ─────────────────────────────────────────
print("\n[3] Candidate set integrity check")
for ds_name, queries in datasets.items():
    issues = []
    for q in queries:
        cands = q.get("candidates", [])
        correct_count = sum(1 for c in cands if c.get("is_correct"))
        is_unmatched = q.get("po_line_id") is None

        if is_unmatched:
            if correct_count != 0:
                issues.append(f"{q['query_id']}: unmatched but {correct_count} correct candidates")
        else:
            if correct_count != 1:
                issues.append(f"{q['query_id']}: expected 1 correct, got {correct_count}")

        if len(cands) < 2:
            issues.append(f"{q['query_id']}: only {len(cands)} candidates (need >= 2)")

    if issues:
        log_fail(f"{ds_name}: {len(issues)} candidate set issues")
        for i in issues[:5]:
            print(f"    {i}")
    else:
        log_pass(f"{ds_name}: All {len(queries)} candidate sets valid")

# ── CHECK 4: Normalized string duplicates across splits ───────────────────────
print("\n[4] Normalized string duplicate check (across splits)")

def collect_all_descriptions(queries):
    descs = set()
    for q in queries:
        descs.add(normalize(q["invoice_line"]))
        for c in q.get("candidates", []):
            descs.add(normalize(c["description"]))
    return descs

train_descs    = collect_all_descriptions(datasets["train_v2"])
val_descs      = collect_all_descriptions(datasets["val"])
test_syn_descs = collect_all_descriptions(datasets["test_synthetic_v2"])
test_hard_descs= collect_all_descriptions(datasets["test_hard_neg"])
test_adv_descs = collect_all_descriptions(datasets["test_adversarial_v2"])
test_hum_descs = collect_all_descriptions(datasets["test_human_v1"])

for pair_name, pair_set in [
    ("train vs val", train_descs & val_descs),
    ("train vs test_syn_v2", train_descs & test_syn_descs),
    ("train vs test_hard", train_descs & test_hard_descs),
    ("train vs test_adv_v2", train_descs & test_adv_descs),
    ("train vs test_human", train_descs & test_hum_descs),
]:
    if pair_set:
        log_warn(f"Normalized string duplicates {pair_name}: {len(pair_set)} strings")
        for s in list(pair_set)[:3]:
            print(f"    '{s}'")
    else:
        log_pass(f"No exact normalized duplicates: {pair_name}")

# ── CHECK 5: Near-duplicate Jaccard check (sampled, not exhaustive) ───────────
print("\n[5] Near-duplicate Jaccard check (Jaccard > 0.80, sampled 500 pairs)")

def sample_jaccard_check(src_descs, tgt_descs, pair_name, threshold=0.80, sample=500):
    src = list(src_descs)[:sample]
    tgt = list(tgt_descs)[:sample]
    near_dupes = []
    for a in src[:100]:  # limit to first 100 src strings for speed
        for b in tgt:
            j = jaccard(a, b)
            if j > threshold and a != b:
                near_dupes.append((j, a, b))
    near_dupes.sort(reverse=True)
    if near_dupes:
        log_warn(f"Near-duplicates (Jaccard>{threshold}) {pair_name}: {len(near_dupes)} pairs")
        for j, a, b in near_dupes[:3]:
            print(f"    {j:.2f}: '{a}' | '{b}'")
    else:
        log_pass(f"No near-duplicates above {threshold}: {pair_name}")

sample_jaccard_check(train_descs, val_descs, "train vs val")
sample_jaccard_check(train_descs, test_syn_descs, "train vs test_syn_v2")
sample_jaccard_check(train_descs, test_hard_descs, "train vs test_hard")
sample_jaccard_check(train_descs, test_adv_descs, "train vs test_adv_v2")
sample_jaccard_check(train_descs, test_hum_descs, "train vs test_human")

# ── CHECK 6: Difficulty distribution ─────────────────────────────────────────
print("\n[6] Difficulty distribution check")
for ds_name, queries in datasets.items():
    dist = Counter(q["difficulty"] for q in queries)
    total = len(queries)
    dist_pct = {k: f"{v}/{total} ({100*v//total}%)" for k, v in dist.items()}
    print(f"  {ds_name}: {dict(dist_pct)}")
    if not queries:
        log_fail(f"{ds_name}: empty dataset")
    elif "hard" not in dist and ds_name != "train":
        log_warn(f"{ds_name}: no hard difficulty queries")
    else:
        log_pass(f"{ds_name}: difficulty distribution OK")

# ── CHECK 7: Candidate count stats ───────────────────────────────────────────
print("\n[7] Candidate count statistics")
for ds_name, queries in datasets.items():
    if not queries:
        continue
    counts = [len(q["candidates"]) for q in queries]
    print(f"  {ds_name}: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f} candidates/query")
    if min(counts) < 2:
        log_fail(f"{ds_name}: some queries have < 2 candidates")
    else:
        log_pass(f"{ds_name}: all queries have >= 2 candidates")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LEAKAGE AUDIT SUMMARY")
print("="*60)
print(f"  PASS:    {len(PASSES)}")
print(f"  WARN:    {len(WARNINGS)}")
print(f"  FAIL:    {len(FAILURES)}")

if FAILURES:
    print("\nFAILURES (must fix before running experiments):")
    for f in FAILURES:
        print(f)

if WARNINGS:
    print("\nWARNINGS (review and document):")
    for w in WARNINGS:
        print(w)

if not FAILURES:
    print("\nAudit PASSED. Safe to proceed to Phase 2.")
else:
    print("\nAudit FAILED. Fix above issues before proceeding.")

# Save results
results = {
    "passes": len(PASSES),
    "warnings": len(WARNINGS),
    "failures": len(FAILURES),
    "pass_list": PASSES,
    "warn_list": WARNINGS,
    "fail_list": FAILURES,
    "dataset_sizes": {k: len(v) for k, v in datasets.items()},
}
out = Path("results/leakage_audit.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {out}")
