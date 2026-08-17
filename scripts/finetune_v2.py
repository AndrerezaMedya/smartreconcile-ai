"""
Fine-tune with improved training data (train_v2.json).
Supports multiple seeds for stability assessment.

Usage:
  python scripts/finetune_v2.py --seed 42
  python scripts/finetune_v2.py --seed 43
  python scripts/finetune_v2.py --seed 44

Outputs:
  models/finetuned_v2_seed{seed}/    — checkpoint
  results/finetuned_v2_seed{seed}_training_log.json
"""

import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--train_file", type=str, default="data/train_v2.json")
args = parser.parse_args()

SEED    = args.seed
EPOCHS  = args.epochs
BATCH   = args.batch
LR      = args.lr
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print(f"Phase 4B: Fine-Tuning v2 — {MODEL_NAME}")
print(f"  Seed={SEED}  Epochs={EPOCHS}  Batch={BATCH}  LR={LR}")
print(f"  Train: {args.train_file}")
print("=" * 60)

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

OUT_MODEL   = MODELS_DIR / f"finetuned_v2_seed{SEED}"
OUT_LOG     = RESULTS_DIR / f"finetuned_v2_seed{SEED}_training_log.json"

# ── Load model ────────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)
print(f"  Embedding dim: {model.get_embedding_dimension()}")

# ── Load training data ────────────────────────────────────────────────────────
with open(args.train_file, encoding="utf-8") as f:
    train_data = json.load(f)

train_examples = []
skipped = 0
for q in train_data:
    inv = q["invoice_line"]
    correct_desc = None
    neg_descs = []
    for c in q["candidates"]:
        if c["is_correct"]:
            correct_desc = c["description"]
        else:
            neg_descs.append(c["description"])

    if correct_desc is None:
        skipped += 1
        continue

    # MNRL: (anchor, positive) pairs + hard negatives via MultipleNegativesRankingLoss
    train_examples.append(InputExample(texts=[inv, correct_desc]))

    # Add one hard neg if available (explicit triplet signals)
    for neg in neg_descs[:1]:
        train_examples.append(InputExample(texts=[inv, correct_desc, neg]))

print(f"  {len(train_examples)} training examples ({skipped} unmatched skipped)")

# ── Validation evaluator ──────────────────────────────────────────────────────
with open(DATA_DIR / "val.json", encoding="utf-8") as f:
    val_data = json.load(f)

val_queries, val_corpus, val_relevant = {}, {}, {}
for q in val_data:
    inv = q["invoice_line"]
    qid = q["query_id"]
    correct_desc = None
    for c in q["candidates"]:
        cid = c["po_line_id"]
        val_corpus[cid] = c["description"]
        if c["is_correct"]:
            correct_desc = cid
    if correct_desc is None:
        continue
    val_queries[qid] = inv
    val_relevant[qid] = {correct_desc}

evaluator = InformationRetrievalEvaluator(
    queries=val_queries,
    corpus=val_corpus,
    relevant_docs=val_relevant,
    name="val",
    mrr_at_k=[1, 3],
    accuracy_at_k=[1, 3],
    precision_recall_at_k=[1, 3],
)

# ── Train ─────────────────────────────────────────────────────────────────────
dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
loss_fn = losses.MultipleNegativesRankingLoss(model)

steps_per_epoch = len(dataloader)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * 0.1))
print(f"  Steps: {total_steps}, Warmup: {warmup_steps}")
print("\nTraining...")

training_log = []

model.fit(
    train_objectives=[(dataloader, loss_fn)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    output_path=str(OUT_MODEL),
    save_best_model=True,
    show_progress_bar=True,
    callback=lambda score, epoch, steps: training_log.append({
        "epoch": epoch, "steps": steps, "val_mrr@1": score
    }),
)

print(f"Best model saved: {OUT_MODEL}")

# Find best epoch
if training_log:
    best = max(training_log, key=lambda x: x["val_mrr@1"])
    print(f"\nBest epoch: {best['epoch']}  val MRR@1: {best['val_mrr@1']:.5f}")

with open(OUT_LOG, "w", encoding="utf-8") as f:
    json.dump({"seed": SEED, "training_log": training_log}, f, indent=2)
print(f"Training log: {OUT_LOG}")
