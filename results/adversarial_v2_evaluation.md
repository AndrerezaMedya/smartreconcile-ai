# Adversarial v2 Evaluation Report

## Dataset Summary
- Total queries: 17 (14 matched + 3 unmatched)
- Categories: 6 (A=spec_trap, B=vendor_SKU, C=near_identical, D=candidate_competition, E=substitution, F=unmatched)

## Per-Category Results (Top-1 Accuracy)

| Category | n | Lexical | Pretrained | FT-v1 | FT-v2 (seed42) |
|----------|---|---------|-----------|-------|---------------|
| spec_trap (A) | 4 | 1.0000 | 0.5000 | 0.5000 | 0.7500 |
| vendor_sku_only (B) | 3 | 1.0000 | 0.0000 | 0.3333 | 0.3333 |
| near_identical (C) | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| candidate_competition (D) | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| substitution (E) | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| unmatched (F) | 3 | N/A | N/A | N/A | N/A |
| **AGGREGATE (matched)** | **14** | **0.9286** | **0.5714** | **0.6429** | **0.7143** |

## Key Findings

### spec_trap (ADV-A, n=4)
Examples: "Pipa Stainless SS316 1 inch Sch 10", "Deep Groove Ball Bearing 6205 2RS", "NYY Cable 4 core 2.5mm2"
- Lexical: 1.000 (PERFECT) — exact token overlap works perfectly for precise spec descriptions
- FT-v2: 0.750 — semantic model correctly handles some (bearing 6205 2RS) but gets confused on similar spec variants
- Root cause FT-v2 failure: "Besi Beton D12 ulir per batang" -> "DEFORMED STEEL BAR D12 GRADE BJTD40" — Indonesian-English cross requires either training or hybrid signal

### vendor_sku_only (ADV-B, n=3)
Examples: "MCB-3P-16A", "SKF-6205-2RS", "BRC M6-150x150"
- Lexical: 1.000 (PERFECT) — token overlap resolves all. "MCB-3P-16A" -> "MCB 3 POLE 16A" works by jaccard.
- Pretrained: 0.000 (TOTAL FAILURE) — embedding space cannot resolve arbitrary vendor codes
- FT-v1, FT-v2: 0.333 — some improvement from training but vendor SKU resolution is fundamentally a lookup problem

### near_identical (ADV-C, n=3)
All models: 1.000 — these are straightforward. "Engine oil SAE 15W-40 API CI-4 drum 209L" maps perfectly.

### candidate_competition (ADV-D, n=2)
All models: 1.000 — "Thermal Overload Relay 4-6A Schneider" and "Roller Chain No. 40 duplex" well-resolved.

### substitution (ADV-E, n=2)
All models: 0.500 — "Hydraulic Hose SAE 100R2AT 1/2 inch 2000 PSI" and "Kunci Pas 17mm"
- One succeeds (kunci pas -> open end wrench, semantic handles this)
- One fails (hydraulic hose spec mapping)

### unmatched (ADV-F, n=3)
"Safety Shoes Jogger Absolute 42", "Biaya Pengiriman / Delivery Fee", "Mur Baut Stainless M8x20 SS316"
- Cannot evaluate top-1 (no correct match in candidate set)
- Used to test false-positive rejection (threshold/margin system)

## Critical Conclusion

**Lexical dominates adversarial v2 (0.929 aggregate) — even on adversarial challenges.**
The main remaining difficulty is "substitution" (0.500 all models), which represents cases where
the invoice item is genuinely different from any PO candidate (semantic gap, not vocabulary gap).

This confirms the primary hypothesis: the architecture should emphasize lexical signals for
exact spec matching, with semantic signals as a complement for vocabulary variation.
