# B3 - Benchmark Validity Audit Report

## Summary
5 synthetic test failures audited. 1 confirmed generator artifact removed from v2.

| Benchmark | Queries | Changes |
|-----------|---------|---------|
| test_synthetic_v1.json | 30 | Original (preserved) |
| test_synthetic_v2.json | 29 | 1 removed (SYN-TE-0013), 1 flagged ambiguous (SYN-TE-0015) |

## Failure Classifications

### SYN-TE-0013 - GENERATOR_ARTIFACT (Removed from v2)
- Invoice: 'Stiker thermal varies'
- Root cause: Generator line 242 builds synonym invoice as '{abbrev} {spec_val[:2]}'. MAT-079 has sizes=['varies'] as catalog placeholder. This caused 'varies' to appear as a spec token.
- Verdict: INVALID. Generator fixed (filter placeholder values).

### SYN-TE-0015 - AMBIGUOUS_LABEL (Retained with flag)
- Invoice: 'Baut' (Indonesian: bolt, general)
- Correct PO: U-BOLT
- Verdict: RETAIN with genuinely_ambiguous=True. Real difficulty: under-specified Indonesian term.

### SYN-TE-0022 - VALID (Real-World Difficulty)
- Invoice: 'BRC'
- Correct PO: WELDED WIRE MESH
- Verdict: VALID. BRC is a genuine industry abbreviation within MVP scope.

### SYN-TE-0024 - VALID (Semantic Neighbor Ambiguity)
- Invoice: 'Rantai' (Indonesian: chain)
- Correct PO: ROLLER CHAIN
- Verdict: VALID. Semantic neighbor confusion (chain vs sprocket).

### SYN-TE-0025 - VALID (Semantic Neighbor Ambiguity)
- Invoice: 'Gear Rantai' (Indonesian: chain gear)
- Correct PO: CHAIN SPROCKET
- Verdict: VALID. Symmetric confusion with SYN-TE-0024 is expected and valid evidence.

## NFR-09 Impact
- v1 (30 queries, 28 matched): lexical = 25/28 = 89.3% (FAIL by 0.7pp)
- v2 (29 queries, 27 matched): lexical = 25/27 = 92.6% (PASS)
- NFR-09 is likely already met by lexical on v2. Verify with full model eval.
