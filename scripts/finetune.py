"""
Phase 4: Fine-Tuning with MultipleNegativesRankingLoss (MNRL)
sentence-transformers v5.7.x — uses SentenceTransformerTrainer API.

Design (per plan §8):
  - MNRL uses in-batch negatives; explicit hard negatives fed via 'negative'
    column in HuggingFace Dataset (NOT triplet loss).
  - Checkpoint: best val cosine_mrr@1 (InformationRetrievalEvaluator)
  - Fixed seed: 42
"""

import json
import random
import re
import warnings
import numpy as np
from pathlib import Path
from datasets import Dataset

# Use non-deprecated import paths
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR  = Path("models/finetuned_minilm")
RESULTS_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS       = 10
BATCH_SIZE   = 16
WARMUP_RATIO = 0.1
LR           = 2e-5


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def jaccard(a: str, b: str) -> float:
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def pick_hard_negative(query_text: str, candidates: list):
    """
    Prefer spec_diff negatives (explicitly curated hard negs), then pick
    the one with highest Jaccard to the query (hardest to distinguish).
    Fallback: any wrong candidate with highest Jaccard.
    """
    wrong = [c for c in candidates if not c["is_correct"]]
    if not wrong:
        return None
    spec_diffs = [c for c in wrong if str(c.get("neg_type", "")).startswith("spec_diff")]
    pool = spec_diffs if spec_diffs else wrong
    pool_scored = sorted(pool, key=lambda c: -jaccard(query_text, c["description"]))
    return pool_scored[0]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_train_dataset(path: Path) -> Dataset:
    """
    Build HuggingFace Dataset with columns: anchor, positive, negative.
    MNRL uses (anchor, positive) pairs with in-batch negatives, plus the
    explicit 'negative' column as an additional hard negative.
    """
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    records = []
    skipped = 0
    for q in queries:
        if q.get("po_line_id") is None:   # unmatched adversarial — skip
            skipped += 1
            continue

        correct_id = q["po_line_id"]
        correct_desc = next(
            (c["description"] for c in q["candidates"] if c["po_line_id"] == correct_id),
            None,
        )
        if correct_desc is None:
            skipped += 1
            continue

        hard_neg = pick_hard_negative(q["invoice_line"], q["candidates"])
        rec = {"anchor": q["invoice_line"], "positive": correct_desc}
        if hard_neg:
            rec["negative"] = hard_neg["description"]
        records.append(rec)

    print(f"  {len(records)} training examples ({skipped} unmatched skipped)")
    return Dataset.from_list(records)


# ── Validation evaluator ──────────────────────────────────────────────────────
def build_ir_evaluator(path: Path, name: str) -> InformationRetrievalEvaluator:
    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    queries_dict  = {}
    corpus_dict   = {}
    relevant_docs = {}

    for q in queries:
        if q.get("po_line_id") is None:
            continue
        qid = q["query_id"]
        queries_dict[qid] = q["invoice_line"]
        relevant_docs[qid] = set()
        for c in q["candidates"]:
            did = f"{qid}_{c['po_line_id']}"
            corpus_dict[did] = c["description"]
            if c["is_correct"]:
                relevant_docs[qid].add(did)

    return InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name=name,
        mrr_at_k=[1, 3],
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        show_progress_bar=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Phase 4: Fine-Tuning — {MODEL_NAME}")
    print(f"  Seed={SEED}  Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LR}")
    print("=" * 60)

    model = SentenceTransformer(MODEL_NAME)
    print(f"  Embedding dim: {model.get_embedding_dimension()}")

    print("\nLoading training data...")
    train_dataset = load_train_dataset(Path("data/train.json"))

    print("Building validation evaluator...")
    val_evaluator = build_ir_evaluator(Path("data/val.json"), name="val")

    # Loss: MNRL — in-batch negatives + explicit hard negative from 'negative' col
    train_loss = MultipleNegativesRankingLoss(model)

    total_steps  = (len(train_dataset) // BATCH_SIZE) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"  Steps: {total_steps}, Warmup: {warmup_steps}")

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        seed=SEED,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_mrr@1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=False,
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=val_evaluator,
    )

    print(f"\nTraining...")
    trainer.train()

    # Best model is already loaded (load_best_model_at_end=True)
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"Best model saved: {OUTPUT_DIR}")

    # Parse training log
    log_history = trainer.state.log_history
    epoch_entries = [e for e in log_history if "eval_val_cosine_mrr@1" in e]
    best = max(epoch_entries, key=lambda e: e.get("eval_val_cosine_mrr@1", 0)) if epoch_entries else {}

    print(f"\nBest epoch: {best.get('epoch')}  val MRR@1: {best.get('eval_val_cosine_mrr@1')}")

    training_log = {
        "model_base": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "mnrl_note": "in-batch negatives + explicit hard negative via 'negative' dataset column",
        "checkpoint_criterion": "eval_val_cosine_mrr@1",
        "best_epoch": best.get("epoch"),
        "best_val_mrr_at_1": best.get("eval_val_cosine_mrr@1"),
        "best_val_acc_at_1": best.get("eval_val_cosine_accuracy@1"),
        "epoch_log": epoch_entries,
    }
    out = RESULTS_DIR / "finetuned_training_log.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    print(f"Training log: {out}")


if __name__ == "__main__":
    main()
