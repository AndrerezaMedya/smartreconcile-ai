# Experiment B — Model Comparison Report

## Full Results Table (Top-1 Accuracy)

| Model | val | syn_v1 | syn_v2 | hard_neg | adv_v2 |
|-------|-----|--------|--------|----------|--------|
| **Lexical** | 0.9643 | 1.0000 | **1.0000** | 0.9459 | **0.9286** |
| Pretrained | 0.6786 | 0.7143 | 0.7407 | 0.8108 | 0.5714 |
| FT-v1 (120ex) | 0.7857 | 0.8214 | 0.8519 | 0.8108 | 0.6429 |
| FT-v2 seed42 | 0.7500 | 0.8214 | 0.8519 | 0.8378 | 0.7143 |
| **FT-v2 seed43** | **0.7857** | **0.8214** | **0.8519** | **0.8378** | **0.7143** |
| **Hybrid (alpha=0.0)** | **0.9643** | **1.0000** | **1.0000** | **0.9459** | **0.9286** |

*Note: Hybrid alpha=0.0 collapses to pure lexical (selected on val only, test was frozen).*

## NFR-09 Status (>=0.90 on test_synthetic_v2)

| Model | Score | NFR-09 |
|-------|-------|--------|
| Lexical | 1.0000 | **PASS** |
| Hybrid | 1.0000 | **PASS** |
| FT-v2 seed43 | 0.8519 | FAIL |
| FT-v1 | 0.8519 | FAIL |
| Pretrained | 0.7407 | FAIL |

## Per-Difficulty Analysis (test_synthetic_v2)

| Difficulty | n | Lexical | FT-v2 | Hybrid |
|-----------|---|---------|-------|--------|
| easy | 6 | 1.000 | 1.000 | 1.000 |
| medium | 13 | 1.000 | 0.923 | 1.000 |
| hard | 7 | 1.000 | 0.571 | 1.000 |
| adversarial | 1 | 1.000 | 1.000 | 1.000 |
| **AGGREGATE** | 27 | **1.000** | **0.852** | **1.000** |

Key: Semantic fails specifically on "hard" difficulty (abbreviations + semantic neighbors).
Lexical/Hybrid succeeds on all difficulties.

## Alpha Sweep Results (on val, frozen test)

Best alpha=0.0 (pure lexical), spec_beta=0.0
Val top-1 vs alpha:
- alpha=0.0: 0.9643 (BEST)
- alpha=0.1: 0.8571 (drops immediately)
- alpha=1.0: 0.7500 (pure semantic)

This definitively shows: on the current dataset distribution, lexical signals are dominant.
Semantic signals do not improve over lexical even as a complement.

## B1 Training Data Impact (ft_v1 vs ft_v2_seed43)

| Dataset | FT-v1 | FT-v2 s43 | Delta |
|---------|-------|----------|-------|
| syn_v2 | 0.8519 | 0.8519 | 0.0000 |
| hard_neg | 0.8108 | 0.8378 | **+0.0270** |
| adv_v2 | 0.6429 | 0.7143 | **+0.0714** |

B1 materially improves adversarial (+7.1pp) and hard_neg (+2.7pp).
Synthetic is unchanged (lexical ceiling dominates).

## Adversarial v2 Per-Category (Lexical vs Best Semantic)

| Category | n | Lexical | FT-v2 s43 |
|----------|---|---------|----------|
| spec_trap | 4 | 1.000 | 0.750 |
| vendor_sku_only | 3 | **1.000** | 0.333 |
| near_identical | 3 | 1.000 | 1.000 |
| candidate_competition | 2 | 1.000 | 1.000 |
| substitution | 2 | 0.500 | 0.500 |
| AGGREGATE | 14 | **0.929** | 0.714 |

Lexical is especially superior on vendor_sku (token overlap resolves arbitrary codes).
All models fail substitution equally (genuine semantic gap).

## Final Model Selection

**Production candidate: Lexical baseline (Jaccard)**
- NFR-09: PASS (1.000 on syn_v2)
- Adversarial: 0.929 (best across all models)
- Simple, fast, interpretable
- No GPU required for inference

**Alternative: Hybrid with alpha=0.0 is identical to lexical**
If future semantic models improve, re-tune alpha on val.
