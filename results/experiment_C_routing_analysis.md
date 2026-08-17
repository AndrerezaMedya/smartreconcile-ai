# Experiment C — Semantic Routing & Usage Analysis

## 1. Overview & Objectives

To verify that the AI system is not a "fake hybrid" (where the model exists in the codebase but never impacts inference), this report quantifies:
1. **Direct Lexical Resolution Rate**: % of queries where lexical signals were confident.
2. **Semantic Routing Rate**: % of queries sent to the fine-tuned semantic backbone.
3. **Semantic Modification Rate**: % of queries where semantic reranking changed the lexical top-1 pick.
4. **Intervention Quality**: Beneficial vs. Harmful vs. Neutral semantic interventions.

---

## 2. Master Routing Breakdown Across All Datasets

| Dataset | Matched Queries | Lexical Direct (%) | Semantic Routed (%) | Semantic Changed (%) | Beneficial (Fixed Error) | Harmful (Created Error) |
|---------|-----------------|--------------------|---------------------|----------------------|--------------------------|-------------------------|
| `val` | 28 | 67.9% | **32.1%** | 10.7% | **+0** | -3 |
| `test_synthetic_v2` | 27 | 77.8% | **22.2%** | 14.8% | **+0** | -4 |
| `test_hard_neg` | 37 | 67.6% | **32.4%** | 13.5% | **+1** | -4 |
| `test_adversarial_v2` | 14 | 71.4% | **28.6%** | 14.3% | **+0** | -2 |
| `test_human_v1` | 18 | 11.1% | **88.9%** | 16.7% | **+1** | -2 |

---

## 3. Qualitative Routing Analysis by Dataset

### A. Human-Written Benchmark (`test_human_v1.json`)
- **Total queries**: 20 (18 matched)
- **Semantic Routing Rate**: **88.9%**
- **Beneficial Interventions**: 1 queries where colloquial Indonesian / abbreviations were resolved by semantic embeddings.

### B. Adversarial Benchmark (`test_adversarial_v2.json`)
- **Total queries**: 17 (14 matched)
- **Semantic Routing Rate**: **28.6%**

### C. Synthetic Benchmark (`test_synthetic_v2.json`)
- **Total queries**: 29 (27 matched)
- **Lexical Direct Rate**: **77.8%**
- **Semantic Routing Rate**: **22.2%**

---

## 4. Query-Level Routing Examples

### Samples from `test_human_v1`

| Query ID | Invoice Line | Action | Lexical Top1 | Chosen PO Line | Correct? |
|----------|--------------|--------|--------------|----------------|----------|
| `HUM-0001` | "Union galv drat 1 inch 150#" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.44 | "STAINLESS STEEL UNION 1 INCH CLASS 150" | ❌ No |
| `HUM-0002` | "Pipa HDPE PE100 3 inch PN16 btg 6 meter" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.27 | "HIGH DENSITY POLYETHYLENE PIPE PE100 3 I" | ✅ Yes |
| `HUM-0003` | "Kanal U UNP 100x50 JIS SS400 pj 6m" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.42 | "U CHANNEL UNP 100X50MM JIS G3101 SS400 L" | ✅ Yes |
| `HUM-0004` | "Kawat BRC M6 spasi 150x150 lembar 2.1x5.4m" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.25 | "WELDED WIRE MESH BRC 6MM OPENING 150X150" | ✅ Yes |
| `HUM-0005` | "Plat kembang tebal 3mm uk 4x8 ft" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.15 | "MILD STEEL FLAT PLATE 3MM 4X8 FEET SS400" | ❌ No |
| `HUM-0006` | "Rantai penggerak roller chain RS 50 simplex box 10ft" | `SEMANTIC_AGREED_WITH_LEXICAL` | 0.46 | "ROLLER CHAIN ANSI 50 SIMPLEX PITCH 15.87" | ✅ Yes |

### Samples from `test_adversarial_v2`

| Query ID | Invoice Line | Action | Lexical Top1 | Chosen PO Line | Correct? |
|----------|--------------|--------|--------------|----------------|----------|
| `ADV-A-001` | "Pipa Stainless Steel SS316 1 inch Sch 10" | `LEXICAL_DIRECT` | 0.78 | "STAINLESS STEEL PIPE SS316 1 INCH SCH 10" | ✅ Yes |
| `ADV-A-002` | "Besi Beton D12 ulir per batang" | `HARMFUL_INTERVENTION` | 0.09 | "PLAIN STEEL BAR D12 GRADE BJTP24" | ❌ No |
| `ADV-A-003` | "Deep Groove Ball Bearing 6205 2RS" | `LEXICAL_DIRECT` | 1.00 | "DEEP GROOVE BALL BEARING 6205 2RS" | ✅ Yes |
| `ADV-A-004` | "NYY Cable 4 core 2.5mm2 0.6/1kV" | `LEXICAL_DIRECT` | 0.90 | "NYY POWER CABLE 4 CORE 2.5MM2 0.6/1KV" | ✅ Yes |
| `ADV-B-001` | "MCB-3P-16A" | `HARMFUL_INTERVENTION` | 0.25 | "MCCB 3 POLE 100A 18KA" | ❌ No |
| `ADV-B-002` | "SKF-6205-2RS" | `LEXICAL_DIRECT` | 0.43 | "DEEP GROOVE BALL BEARING 6205 2RS SKF" | ✅ Yes |

### Samples from `val`

| Query ID | Invoice Line | Action | Lexical Top1 | Chosen PO Line | Correct? |
|----------|--------------|--------|--------------|----------------|----------|
| `VAL-VA-0001` | "PVC PIPE N" | `LEXICAL_DIRECT` | 0.50 | "PVC PIPE 2" | ✅ Yes |
| `VAL-VA-0002` | "HYDRAULIC HOSE / L= " | `LEXICAL_DIRECT` | 0.75 | "HYDRAULIC HOSE   L=M" | ✅ Yes |
| `VAL-VA-0003` | "c channel / c purlin 1 t=." | `LEXICAL_DIRECT` | 0.67 | "C CHANNEL / C PURLIN 5 T=." | ✅ Yes |
| `VAL-VA-0004` | "BARBED WIRE" | `LEXICAL_DIRECT` | 1.00 | "BARBED WIRE" | ✅ Yes |
| `VAL-VA-0005` | "V-BELT" | `LEXICAL_DIRECT` | 1.00 | "V-BELT" | ✅ Yes |
| `VAL-VA-0006` | "OIL SEAL / ROTARY SHAFT SEAL" | `LEXICAL_DIRECT` | 1.00 | "OIL SEAL / ROTARY SHAFT SEAL" | ✅ Yes |

