# SmartReconcile AI

> **Enterprise Invoice-to-PO Reconciliation & Commercial Verification Engine**  
> *COMPFEST 18 — AI Innovation Challenge (AIC)*  
> **Theme**: AI for the Backbone of the Economy (Smart Manufacturing & Smart Commerce)

---

## 1. Overview

**SmartReconcile AI** is an enterprise Accounts Payable (AP) reconciliation engine designed for Indonesian B2B supply chains and manufacturing operations. It resolves a pervasive economic inefficiency: line-item discrepancies between billed supplier invoices and contracted Purchase Orders (POs) caused by trade abbreviations, non-standard catalog descriptions, specification traps, unit-of-measure mismatches, and price variances.

### Core Operating Principle
$$\text{AI Recommends} \longrightarrow \text{Rules Verify} \longrightarrow \text{Humans Decide} \longrightarrow \text{Commit to Ledger}$$

The system enforces strict financial governance:
- **Predictive AI** generates semantic matching recommendations and ranked PO alternatives.
- **Deterministic Rule Verification** independently calculates quantity overbilling, unit price variances, UOM compatibility, and line arithmetic.
- **Human AP Reviewers** retain sole decision authority to approve, override, or dispute exceptions.
- **Zero AI Auto-Commits**: Financial commitments to the permanent ledger require explicit human authorization.

---

## 2. System Architecture

```
                                  [ Supplier Invoice ]
                                (Digital PDF / CSV / JSON)
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ 1. INGESTION & EXTRACTION       │
                           │    pdfplumber / Header & Lines  │
                           └─────────────────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ 2. HYBRID MATCHING ENGINE       │
                           │    • Lexical Gating (Jaccard)   │
                           │    • Fine-Tuned MiniLM-L12-v2   │
                           │    • Greedy 1:1 Mutual Excl.    │
                           └─────────────────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ 3. DETERMINISTIC 4-WAY VERIFIER │
                           │    [QTY]  [PRICE]  [UOM]  [MATH]│
                           └─────────────────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ 4. HUMAN REVIEW WORKSPACE       │
                           │    Side-by-Side Review & Notes  │
                           │    Approve / Override / Dispute │
                           └─────────────────────────────────┘
                                            │
                                            ▼
                           ┌─────────────────────────────────┐
                           │ 5. COMMITTED RECONCILIATION     │
                           │    Audit Trail & Permanent DB   │
                           └─────────────────────────────────┘
```

### Key Technical Pillars

1. **Hybrid Lexical + Semantic Matcher (`app/core/matcher.py`)**:
   - **Lexical Gating**: Computes token-level Jaccard similarity. Unambiguous exact matches bypass heavy embedding overhead.
   - **Fine-Tuned Semantic Model**: When lexical confidence is ambiguous ($\text{margin} \le 0.10$ or $\text{top1} \le 0.15$), queries activate the domain-fine-tuned multilingual model (`models/finetuned_v2_seed44`).
   - **Greedy 1:1 Mutual Exclusivity**: Ensures no single PO line item is assigned to multiple invoice lines.

2. **Deterministic 4-Way Commercial Verifier (`app/core/verifier.py`)**:
   - **Quantity Verification**: Flags overbilling beyond contracted PO quantity.
   - **Price Variance Verification**: Enforces a $\le 1.0\%$ commercial tolerance threshold.
   - **UOM Compatibility**: Normalizes trade units (e.g., *batang* vs *meter*, *roll* vs *meter*, *box* vs *pcs*).
   - **Line Arithmetic & Rounding**: Verifies $\text{Qty} \times \text{Unit Price} = \text{Line Total}$ within $\pm 1.0\text{ IDR}$.

3. **Enterprise Web Console (`app/ui/`)**:
   - High-density operational workspace built with pure native ES6 modules and semantic CSS.
   - 100% synchronous offline architecture with ~137ms average CPU inference latency.

---

## 3. Machine Learning Model & Provenance

- **Backbone Architecture**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Fine-Tuning Methodology**: Multiple Negatives Ranking Loss (MNRL) trained on 148 Indonesian industrial catalog triplets (Anchors, Positives, and Hard Negatives).
- **Target Invariant**: High sensitivity to specification grades (e.g., distinguishing SS304 vs SS316, Sch 40 vs Sch 80) while robust to trade abbreviations (e.g., *siku tbl*, *pipa bsi*).
- **Runtime Model Path**: `models/finetuned_v2_seed44/` (465.15 MB total disk footprint).
- **Inference Runtime**: 100% In-Memory CPU execution (Zero GPU/Cloud API requirements).

---

## 4. Repository Structure

```
smartreconcile-ai/
├── app/
│   ├── main.py                  # Application entrypoint (Starlette)
│   ├── api/
│   │   ├── routes.py            # REST API endpoints
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── core/
│   │   ├── config.py            # Frozen hyperparameters & paths
│   │   ├── matcher.py           # Hybrid matcher & greedy 1:1 assignment
│   │   ├── models.py            # Domain data structures
│   │   ├── reconciler.py        # Core reconciliation orchestrator
│   │   └── verifier.py          # Deterministic 4-way commercial rules
│   ├── db/
│   │   ├── database.py          # SQLite connection & schema initialization
│   │   ├── po_repository.py     # Purchase order database access
│   │   └── staging_repository.py# Staging & committed reconciliation storage
│   ├── services/
│   │   ├── extractor.py         # PDF/CSV/JSON invoice data extraction
│   │   ├── staging_service.py   # Staging lifecycle & commit management
│   │   └── validator.py         # File format & size pre-validation
│   └── ui/
│       ├── templates/
│       │   └── index.html       # Single-page enterprise workspace
│       └── static/
│           ├── css/style.css    # Enterprise dark-slate design system
│           └── js/
│               ├── app.js       # Application orchestrator
│               └── modules/     # Native ES6 modules (DAG architecture)
├── data/
│   ├── reconcile_erp.db         # Pre-seeded SQLite master database
│   ├── material_catalog.json    # Master material catalog reference data
│   ├── demo_invoices_pdf/       # 24 digital test PDF invoice fixtures
│   ├── train_v2.json / val.json # Fine-tuning dataset splits
│   └── workflow_simulation_invoices.json # Multi-line benchmark scenario fixtures
├── models/
│   └── finetuned_v2_seed44/     # Validated runtime weights & tokenizer (Git LFS)
├── results/                     # Empirical validation & ablation reports
├── scripts/                     # Evaluation, simulation & benchmark scripts
├── tests/                       # Automated unit & integration test suite
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Compose orchestration
├── requirements.txt             # Python dependency manifest
├── .gitattributes               # Git LFS tracking rules
├── .gitignore                   # Repository exclusion rules
├── .dockerignore                # Docker build-context exclusion rules
└── README.md                    # System documentation
```

---

## 5. Getting Started

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git & Git LFS (`git lfs install`)
- *Optional*: Docker & Docker Compose

---

### Option A: Local Python Environment

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AndrerezaMedya/smartreconcile-ai.git
   cd smartreconcile-ai
   git lfs pull
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   
   # Linux/macOS:
   source .venv/bin/activate
   
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch Application**:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

5. **Access Console**:
   Open browser at `http://127.0.0.1:8000/`.

---

### Option B: Docker Compose (Single Command)

1. **Build and Run Container**:
   ```bash
   docker compose up --build
   ```

2. **Access Console**:
   Open browser at `http://localhost:8000/`.

---

## 6. Running Automated Tests

Run the complete 21-test suite covering core matching, verifier rules, file extraction, API endpoints, and staging workflows:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Expected Output**:
```text
Ran 21 tests in ~4.0s
OK
```

---

## 7. Operational Workflow & Demo Guide

1. **Reconciliation Workspace**:
   - Filter through 24 pre-generated test scenarios across 5 business categories (*Standard Clean*, *Indo Trade Terms*, *Spec Traps*, *Discrepancies*, *Exceptions*).
   - Inspect the high-density 3-column table showing Billed Item, AI Match Routing & Confidence, and Matched PO Line with 4-Way Verification chips (`QTY`, `PRICE`, `UOM`, `MATH`).

2. **Human Review Decision Modal**:
   - Click `[ Resolve ]` on exception lines (e.g. `INV-2026-017` Price Variance or `INV-2026-011` Spec Trap).
   - Review side-by-side numerical deltas, inspect AI confidence margins vs rule failures, select alternative candidate PO lines, input reviewer authorization notes, and confirm decisions.

3. **Enterprise Ingestion Pipeline**:
   - Navigate to `Upload Invoice`.
   - Upload any digital PDF, CSV, or JSON invoice (or click any Quick Demo scenario card).
   - Watch the 4-stage pipeline transition (`Upload` $\to$ `Validate` $\to$ `Extract` $\to$ `AI Match`) and review extracted line metrics.

4. **Commit Checkpoint & Ledger**:
   - Click `[ Commit Reconciliation ]` on resolved staging records.
   - Inspect financial verification totals and reviewer declaration.
   - Confirm commit to write permanently to SQLite ledger and inspect audit event logs in `Compliance Audit`.

5. **Batch Simulation Benchmark**:
   - Click `[ Batch Benchmark (24) ]` in the top navigation bar to execute all 24 scenarios (106 line items) in CPU memory.

---

## 8. License

Developed for COMPFEST 18 — AI Innovation Challenge (AIC).
All intellectual property remains with the development team in accordance with competition rules.
