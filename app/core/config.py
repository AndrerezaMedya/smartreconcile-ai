"""
Configuration settings and frozen hyperparameters for the Reconciliation MVP.
All values are strictly based on frozen experimental validations from Phases 1–9.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Validated Model Weights
DEFAULT_MODEL_PATH = MODELS_DIR / "finetuned_v2_seed44"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = MODELS_DIR / "finetuned_v2_seed43"

# Frozen Gating Trigger Thresholds (from Experiment C)
TAU_MARGIN = 0.10          # Lexical margin threshold below which query is considered ambiguous
TAU_TOP1_SCORE = 0.15      # Lexical top-1 score threshold below which overlap is considered low
SEMANTIC_BLEND_ALPHA = 0.40 # Semantic weight when blended scoring is activated

# Frozen Confidence Gating Thresholds
CONFIDENCE_MARGIN_THRESHOLD = 0.15 # Margin required for MATCHED status without human review
MIN_SIMILARITY_FLOOR = 0.15        # Minimum similarity required for assignability

# Deterministic Numeric Tolerances (SAP standard)
PRICE_TOLERANCE_PCT = 0.01 # 1% commercial price tolerance
MATH_TOLERANCE_IDR = 1.0   # 1 IDR tolerance for rounding

# Application Settings
APP_TITLE = "SmartReconcile AI — Invoice-to-PO Reconciliation"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-assisted line-item matching and 4-way commercial verification for B2B procurement."
