"""
Phase 1: Dataset Generator
Generates candidate ranking sets for invoice-to-PO matching benchmark.

Output format per query:
{
  "query_id": "...",
  "invoice_line": "...",       # simulates vendor's invoice description
  "po_line_id": "...",         # the correct PO line id
  "candidates": [
    {"po_line_id": "...", "description": "...", "is_correct": bool, "neg_type": "..."}
  ],
  "difficulty": "easy|medium|hard|adversarial",
  "variation_types": [...],
  "material_id": "...",
  "category": "..."
}
"""

import json
import random
import re
import itertools
from pathlib import Path
from typing import Optional

random.seed(42)

# ── Load catalog ──────────────────────────────────────────────────────────────
CATALOG_PATH = Path("data/material_catalog.json")
with open(CATALOG_PATH, encoding="utf-8") as f:
    CATALOG = json.load(f)

MATERIALS = {m["material_id"]: m for m in CATALOG["materials"]}
BY_SPLIT = {
    "train": [m for m in CATALOG["materials"] if m["split"] == "train"],
    "validation": [m for m in CATALOG["materials"] if m["split"] == "validation"],
    "test": [m for m in CATALOG["materials"] if m["split"] == "test"],
}
BY_CATEGORY = {}
for m in CATALOG["materials"]:
    BY_CATEGORY.setdefault(m["category"], []).append(m["material_id"])


# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTION GENERATORS
# Each function returns a string description in a specific style.
# ══════════════════════════════════════════════════════════════════════════════

def _pick(lst):
    return random.choice(lst) if lst else ""

def _maybe(val, prob=0.5):
    """Return val with probability prob, else None."""
    return val if random.random() < prob else None


# ── Style A: ERP / PO style (uppercase, formal, complete) ─────────────────────
def style_po(mat: dict, spec_override: dict = None) -> str:
    """PO-style: UPPERCASE, full words, formal spec."""
    spec = {**(mat.get("specifications") or {}), **(spec_override or {})}
    name = mat["canonical_name_en"].upper()
    parts = [name]

    cat = mat["category"]
    if cat == "piping":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]).upper())
        if "schedules" in spec: parts.append(_pick(spec["schedules"]).upper())
        if "lengths" in spec: parts.append(f"L={_pick(spec['lengths']).upper()}")
    elif cat == "structural_steel":
        if "diameters" in spec: parts.append(_pick(spec["diameters"]).upper())
        if "sizes" in spec: parts.append(_pick(spec["sizes"]).upper())
        if "thickness" in spec: parts.append(f"T={_pick(spec['thickness'])}")
        if "lengths" in spec: parts.append(_pick(spec["lengths"]).upper())
    elif cat == "bearings":
        if "numbers" in spec: parts.append(_pick(spec["numbers"]))
        if "seals" in spec:
            seal = _pick(spec["seals"])
            parts.append(seal.split(" ")[0])  # e.g., "2RS"
    elif cat == "electrical":
        if "current_rating" in spec: parts.append(_pick(spec["current_rating"]))
        if "poles" in spec: parts.append(_pick(spec["poles"]))
        if "cores" in spec: parts.append(_pick(spec["cores"]).upper())
        if "cross_section" in spec: parts.append(_pick(spec["cross_section"]))
        if "wattage" in spec: parts.append(_pick(spec["wattage"]))
    elif cat == "chemicals":
        if "grades" in spec: parts.append(_pick(spec["grades"]))
        if "packaging" in spec: parts.append(_pick(spec["packaging"]).upper())
    elif cat == "packaging":
        if "sizes" in spec and spec["sizes"] != "varies per product": parts.append(_pick(spec["sizes"]))
        if "thickness" in spec: parts.append(_pick(spec["thickness"]))
    elif cat == "fasteners":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]).upper())
        if "grades" in spec: parts.append(f"GR.{_pick(spec['grades'])}")
        if "lengths" in spec: parts.append(_pick(spec["lengths"]))
    elif cat == "tools_consumables":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
        if "standards" in spec: parts.append(_pick(spec["standards"]))

    return " ".join(p for p in parts if p)


# ── Style B: Vendor invoice style (Indonesian, mixed case, abbreviated) ────────
def style_invoice_id(mat: dict, spec_override: dict = None) -> str:
    """Invoice style: Indonesian name, mixed case, abbreviations common."""
    spec = {**(mat.get("specifications") or {}), **(spec_override or {})}
    abbrevs = mat.get("common_abbreviations", [])
    name = _pick(abbrevs) if abbrevs and random.random() < 0.6 else mat["canonical_name"]
    parts = [name]

    cat = mat["category"]
    if cat == "piping":
        if "sizes" in spec:
            sz = _pick(spec["sizes"])
            # vary the format
            sz = sz.replace(" inch", '"').replace("inch", '"') if random.random() < 0.5 else sz
            parts.append(sz)
        if "schedules" in spec and random.random() < 0.6:
            parts.append(_pick(spec["schedules"]).replace(" ", "").lower())
        if "lengths" in spec and random.random() < 0.5:
            l = _pick(spec["lengths"])
            parts.append(f"p.{l}" if random.random() < 0.3 else l)
    elif cat == "structural_steel":
        if "diameters" in spec: parts.append(_pick(spec["diameters"]))
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
        if "thickness" in spec and random.random() < 0.5:
            t = _pick(spec["thickness"])
            parts.append(f"t={t}" if random.random() < 0.4 else t)
    elif cat == "bearings":
        if "numbers" in spec: parts.append(_pick(spec["numbers"]))
        if "seals" in spec and random.random() < 0.7:
            seal = _pick(spec["seals"])
            parts.append(seal.split(" ")[0])
    elif cat == "electrical":
        if "current_rating" in spec: parts.append(_pick(spec["current_rating"]))
        if "poles" in spec and random.random() < 0.5: parts.append(_pick(spec["poles"]))
        if "cores" in spec: parts.append(_pick(spec["cores"]))
        if "cross_section" in spec: parts.append(_pick(spec["cross_section"]))
        if "wattage" in spec: parts.append(_pick(spec["wattage"]))
    elif cat == "chemicals":
        if "grades" in spec: parts.append(_pick(spec["grades"]))
        if "packaging" in spec and random.random() < 0.5:
            parts.append(_pick(spec["packaging"]))
    elif cat == "fasteners":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
        if "lengths" in spec and random.random() < 0.4: parts.append(_pick(spec["lengths"]))
    elif cat == "tools_consumables":
        if "sizes" in spec and random.random() < 0.7: parts.append(_pick(spec["sizes"]))

    return " ".join(p for p in parts if p)


# ── Style C: Short / truncated (common on old systems with char limits) ────────
def style_truncated(mat: dict) -> str:
    full = style_invoice_id(mat)
    words = full.split()
    keep = max(1, len(words) - random.randint(1, 2))
    return " ".join(words[:keep])


# ── Style D: English-heavy vendor style ───────────────────────────────────────
def style_english(mat: dict, spec_override: dict = None) -> str:
    spec = {**(mat.get("specifications") or {}), **(spec_override or {})}
    name = mat["canonical_name_en"]
    parts = [name]

    cat = mat["category"]
    if cat == "piping":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
        if "schedules" in spec: parts.append(_pick(spec["schedules"]))
    elif cat == "structural_steel":
        if "diameters" in spec: parts.append(_pick(spec["diameters"]))
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
    elif cat == "bearings":
        if "numbers" in spec: parts.append(_pick(spec["numbers"]))
        if "seals" in spec: parts.append(_pick(spec["seals"]).split(" ")[0])
    elif cat == "chemicals":
        if "grades" in spec: parts.append(_pick(spec["grades"]))
    elif cat == "fasteners":
        if "sizes" in spec: parts.append(_pick(spec["sizes"]))
        if "lengths" in spec and random.random() < 0.5: parts.append(_pick(spec["lengths"]))
    elif cat == "electrical":
        if "current_rating" in spec: parts.append(_pick(spec["current_rating"]))
        if "cores" in spec: parts.append(_pick(spec["cores"]))
        if "cross_section" in spec: parts.append(_pick(spec["cross_section"]))
    elif cat == "tools_consumables":
        if "sizes" in spec and random.random() < 0.7: parts.append(_pick(spec["sizes"]))

    return " ".join(p for p in parts if p)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY GENERATORS — by difficulty tier
# ══════════════════════════════════════════════════════════════════════════════

def _shared_spec(mat: dict) -> dict:
    """Pick ONE consistent spec for a query (so invoice & PO describe same item)."""
    spec_choices = {}
    raw = mat.get("specifications", {})
    for k, v in raw.items():
        if isinstance(v, list) and v:
            spec_choices[k] = _pick(v)
    return spec_choices


def make_easy_query(mat: dict, qid: str) -> dict:
    """Easy: both invoice and PO describe the same item in similar styles."""
    spec = _shared_spec(mat)
    po_desc = style_po(mat, spec)
    invoice_desc = style_po(mat, spec)  # same style, nearly identical
    # Minor variation: lowercase invoice
    if random.random() < 0.5:
        invoice_desc = invoice_desc.lower()

    correct = {"po_line_id": f"{qid}-PL-C", "description": po_desc,
               "is_correct": True, "neg_type": None}
    negatives = _pick_negatives(mat, qid, n=3, style_fn=style_po, spec=spec)

    return {
        "query_id": qid,
        "invoice_line": invoice_desc,
        "po_line_id": f"{qid}-PL-C",
        "candidates": _shuffle([correct] + negatives),
        "difficulty": "easy",
        "variation_types": ["near_exact"],
        "material_id": mat["material_id"],
        "category": mat["category"],
    }


def make_medium_query(mat: dict, qid: str, variation: str) -> dict:
    """Medium: invoice uses abbreviations, synonyms, word reorder, or ID/EN mix."""
    spec = _shared_spec(mat)
    po_desc = style_po(mat, spec)

    if variation == "abbreviation":
        invoice_desc = style_invoice_id(mat, spec)
        vtypes = ["abbreviation"]
    elif variation == "synonym":
        abbrevs = mat.get("common_abbreviations", [])
        if abbrevs:
            _PLACEHOLDERS = {"varies", "n/a", "tbd", "-", "custom", "variable", "various"}
            spec_vals = [str(v) for v in list(spec.values())[:2]
                         if str(v).lower() not in _PLACEHOLDERS]
            invoice_desc = " ".join([_pick(abbrevs)] + spec_vals)
        else:
            invoice_desc = style_invoice_id(mat, spec)
        vtypes = ["synonym"]
    elif variation == "word_reorder":
        words = style_po(mat, spec).split()
        if len(words) > 2:
            random.shuffle(words)
        invoice_desc = " ".join(words)
        vtypes = ["word_reorder"]
    elif variation == "truncation":
        invoice_desc = style_truncated(mat)
        vtypes = ["truncation"]
    elif variation == "lang_mix":
        invoice_desc = style_english(mat, spec)
        vtypes = ["indonesian_english_mix"]
    else:
        invoice_desc = style_invoice_id(mat, spec)
        vtypes = ["abbreviation"]

    correct = {"po_line_id": f"{qid}-PL-C", "description": po_desc,
               "is_correct": True, "neg_type": None}
    negatives = _pick_negatives(mat, qid, n=3, style_fn=style_po, spec=spec)

    return {
        "query_id": qid,
        "invoice_line": invoice_desc,
        "po_line_id": f"{qid}-PL-C",
        "candidates": _shuffle([correct] + negatives),
        "difficulty": "medium",
        "variation_types": vtypes,
        "material_id": mat["material_id"],
        "category": mat["category"],
    }


def make_hard_query(mat: dict, qid: str) -> Optional[dict]:
    """Hard: spec-diff hard negatives. Invoice and correct PO share same material
    family but differ only on spec (size, grade, seal, etc.)."""
    spec = _shared_spec(mat)
    po_desc = style_po(mat, spec)
    invoice_desc = style_invoice_id(mat, spec)

    hard_negs = _pick_spec_diff_negatives(mat, qid, spec, n=2)
    if not hard_negs:
        return None  # Can't make a hard query for this material

    easy_negs = _pick_negatives(mat, qid, n=1, style_fn=style_po, spec=spec)
    correct = {"po_line_id": f"{qid}-PL-C", "description": po_desc,
               "is_correct": True, "neg_type": None}

    return {
        "query_id": qid,
        "invoice_line": invoice_desc,
        "po_line_id": f"{qid}-PL-C",
        "candidates": _shuffle([correct] + hard_negs + easy_negs),
        "difficulty": "hard",
        "variation_types": ["spec_diff_hard_negative"],
        "material_id": mat["material_id"],
        "category": mat["category"],
    }


def make_adversarial_query(mat: dict, qid: str, adv_type: str) -> Optional[dict]:
    """Adversarial: missing PO line, invoice-only, substituted item, competing candidates."""
    spec = _shared_spec(mat)

    if adv_type == "unmatched":
        # Invoice line has NO correct match in candidates
        invoice_desc = style_invoice_id(mat, spec)
        negatives = _pick_negatives(mat, qid, n=4, style_fn=style_po, spec=spec)
        if not negatives:
            return None
        return {
            "query_id": qid,
            "invoice_line": invoice_desc,
            "po_line_id": None,  # no correct match
            "candidates": _shuffle(negatives),
            "difficulty": "adversarial",
            "variation_types": ["unmatched_line"],
            "material_id": mat["material_id"],
            "category": mat["category"],
        }

    elif adv_type == "vendor_sku":
        # Invoice only has a short SKU-like code, no text description
        abbrevs = mat.get("common_abbreviations", [])
        sku_parts = []
        if abbrevs:
            short = abbrevs[0][:4].upper()
            sku_parts.append(short)
        spec_vals = list(spec.values())
        if spec_vals:
            sku_parts.append(str(spec_vals[0]).replace(" ", "").upper()[:8])
        invoice_desc = "-".join(sku_parts) if sku_parts else "SKU-UNKNOWN"

        po_desc = style_po(mat, spec)
        correct = {"po_line_id": f"{qid}-PL-C", "description": po_desc,
                   "is_correct": True, "neg_type": None}
        negatives = _pick_negatives(mat, qid, n=3, style_fn=style_po, spec=spec)
        return {
            "query_id": qid,
            "invoice_line": invoice_desc,
            "po_line_id": f"{qid}-PL-C",
            "candidates": _shuffle([correct] + negatives),
            "difficulty": "adversarial",
            "variation_types": ["vendor_sku_only"],
            "material_id": mat["material_id"],
            "category": mat["category"],
        }

    return None


# ══════════════════════════════════════════════════════════════════════════════
# NEGATIVE SAMPLERS
# ══════════════════════════════════════════════════════════════════════════════

def _pick_negatives(mat: dict, qid: str, n: int, style_fn, spec: dict) -> list:
    """Pick n 'different item' negatives from same split."""
    candidates = []
    split = mat["split"]
    mat_id = mat["material_id"]

    # First try same-category materials
    same_cat = [m for m in BY_SPLIT[split]
                if m["material_id"] != mat_id
                and m["category"] == mat["category"]]
    random.shuffle(same_cat)

    # Then cross-category
    diff_cat = [m for m in BY_SPLIT[split]
                if m["material_id"] != mat_id
                and m["category"] != mat["category"]]
    random.shuffle(diff_cat)

    pool = same_cat + diff_cat

    for i, neg_mat in enumerate(pool[:n]):
        neg_spec = _shared_spec(neg_mat)
        desc = style_fn(neg_mat, neg_spec)
        neg_type = ("same_category" if neg_mat["category"] == mat["category"]
                    else "different_item")
        candidates.append({
            "po_line_id": f"{qid}-PL-N{i+1}",
            "description": desc,
            "is_correct": False,
            "neg_type": neg_type,
        })

    return candidates


def _pick_spec_diff_negatives(mat: dict, qid: str, shared_spec: dict, n: int) -> list:
    """
    Pick hard negatives that are the SAME material but with a DIFFERENT spec value.
    E.g., BEARING 6205 2RS vs BEARING 6205 ZZ, or PIPE 2" vs PIPE 3"
    """
    raw = mat.get("specifications", {})
    results = []
    idx = 0

    # Try each spec field that has multiple options
    for key, values in raw.items():
        if not isinstance(values, list) or len(values) < 2:
            continue
        chosen_val = shared_spec.get(key)
        if chosen_val is None:
            continue
        alts = [v for v in values if v != chosen_val]
        if not alts:
            continue

        alt_val = _pick(alts)
        alt_spec = {**shared_spec, key: alt_val}
        desc = style_po(mat, alt_spec)
        results.append({
            "po_line_id": f"{qid}-PL-HN{idx+1}",
            "description": desc,
            "is_correct": False,
            "neg_type": f"spec_diff_{key}",
        })
        idx += 1
        if idx >= n:
            break

    return results


def _shuffle(lst: list) -> list:
    """Shuffle candidates and deduplicate by description (keep first seen, always keep correct)."""
    lst = list(lst)
    # Separate correct candidate (must always be kept)
    correct_cands = [c for c in lst if c.get("is_correct")]
    other_cands   = [c for c in lst if not c.get("is_correct")]

    # Deduplicate negatives by normalized description
    seen_descs = {c["description"].strip().upper() for c in correct_cands}
    deduped_negs = []
    for c in other_cands:
        key = c["description"].strip().upper()
        if key not in seen_descs:
            seen_descs.add(key)
            deduped_negs.append(c)

    result = correct_cands + deduped_negs
    random.shuffle(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_split(split_name: str, target_counts: dict, prefix: str = None) -> list:
    """
    target_counts: {
        "easy": N, "medium": N, "hard": N, "adversarial": N
    }
    prefix: override the auto-generated prefix to ensure unique query_ids across datasets
    """
    materials = BY_SPLIT[split_name]
    queries = []
    counter = {"easy": 0, "medium": 0, "hard": 0, "adversarial": 0}
    qid_counter = [0]

    if prefix is None:
        prefix = split_name[:3].upper()

    def next_qid(pfx):
        qid_counter[0] += 1
        return f"{pfx}-{split_name[:2].upper()}-{qid_counter[0]:04d}"

    medium_variations = ["abbreviation", "synonym", "word_reorder", "truncation", "lang_mix"]
    adv_types = ["unmatched", "vendor_sku"]

    # Cycle through materials, generating queries per difficulty
    mat_cycle = itertools.cycle(materials)

    # Easy
    for _ in range(target_counts.get("easy", 0)):
        mat = next(mat_cycle)
        q = make_easy_query(mat, next_qid(prefix))
        queries.append(q)
        counter["easy"] += 1

    # Medium
    var_cycle = itertools.cycle(medium_variations)
    for _ in range(target_counts.get("medium", 0)):
        mat = next(mat_cycle)
        variation = next(var_cycle)
        q = make_medium_query(mat, next_qid(prefix), variation)
        queries.append(q)
        counter["medium"] += 1

    # Hard
    attempts = 0
    while counter["hard"] < target_counts.get("hard", 0) and attempts < 500:
        mat = next(mat_cycle)
        q = make_hard_query(mat, next_qid(prefix))
        if q:
            queries.append(q)
            counter["hard"] += 1
        attempts += 1

    # Adversarial
    adv_cycle = itertools.cycle(adv_types)
    for _ in range(target_counts.get("adversarial", 0)):
        mat = next(mat_cycle)
        adv_type = next(adv_cycle)
        q = make_adversarial_query(mat, next_qid(prefix), adv_type)
        if q:
            queries.append(q)
            counter["adversarial"] += 1

    print(f"  [{split_name}] Generated: {counter}")
    return queries


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # ── Training split ──────────────────────────────────────────────────────
    print("Generating train split...")
    train = generate_split("train", {
        "easy": 24,      # 20% — model must not regress on these
        "medium": 60,    # 50% — core training signal
        "hard": 24,      # 20% — hard negative training
        "adversarial": 12,  # 10% — edge cases
    })  # target: 120

    # ── Validation split ────────────────────────────────────────────────────
    print("Generating validation split...")
    val = generate_split("validation", {
        "easy": 6,
        "medium": 14,
        "hard": 7,
        "adversarial": 3,
    })  # target: 30

    # ── Test splits ─────────────────────────────────────────────────────────
    print("Generating test_synthetic split...")
    test_syn = generate_split("test", {
        "easy": 6,
        "medium": 14,
        "hard": 7,
        "adversarial": 3,
    }, prefix="SYN")  # target: 30

    # Hard-negative test (manually curated via catalog, same generator but hard-only)
    print("Generating test_hard_neg split...")
    test_hard = generate_split("test", {
        "easy": 5,
        "medium": 10,
        "hard": 20,    # heavier hard-neg weighting
        "adversarial": 5,
    }, prefix="HRD")  # target: 40

    # ── Save ─────────────────────────────────────────────────────────────────
    files = {
        "train.json": train,
        "val.json": val,
        "test_synthetic.json": test_syn,
        "test_hard_neg.json": test_hard,
    }
    for fname, data in files.items():
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved {out_path} ({len(data)} queries)")

    print("\nNote: test_human.json must be created manually.")
    print("Done.")

    # Quick stats
    print("\n── Summary ──────────────────────────────────────────────────")
    for fname, data in files.items():
        diffs = {}
        for q in data:
            diffs[q["difficulty"]] = diffs.get(q["difficulty"], 0) + 1
        total_candidates = sum(len(q["candidates"]) for q in data)
        print(f"  {fname}: {len(data)} queries, {total_candidates} total pairs | {diffs}")


if __name__ == "__main__":
    main()
