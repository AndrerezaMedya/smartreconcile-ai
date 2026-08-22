"""
Pydantic data models and schemas for Invoice-to-PO Reconciliation.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ReconciliationStatus(str, Enum):
    MATCHED = "MATCHED"          # High confidence, 0 discrepancies
    AMBIGUOUS = "AMBIGUOUS"      # Needs human review (low margin or discrepancy flag)
    UNMATCHED = "UNMATCHED"      # No candidate or out-of-catalog exception


class ReviewActionType(str, Enum):
    APPROVE = "APPROVE"          # Accept current recommendation
    OVERRIDE = "OVERRIDE"        # Reassign to different candidate PO line
    REJECT = "REJECT"            # Mark line as invalid / credit memo required


# ── Ingestion Schemas ─────────────────────────────────────────────────────────

class InvoiceLine(BaseModel):
    line_no: int
    description: str
    qty: float
    uom: str
    unit_price: float
    line_total: float
    gold_po_line_no: Optional[int] = None
    expected_flag: Optional[str] = None


class Invoice(BaseModel):
    # invoice_id is the document's identity and is required: a document we cannot
    # identify cannot be staged or audited. Every other header field is nullable
    # (FR-08) — a field that could not be extracted stays None rather than being
    # filled with an invented value.
    invoice_id: str
    po_id: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_date: Optional[str] = None
    scenario_category: Optional[str] = "custom"
    scenario_description: Optional[str] = ""
    invoice_lines: List[InvoiceLine]


class POLine(BaseModel):
    po_line_no: int
    description: str
    ordered_qty: float
    uom: str
    unit_price: float
    line_total: Optional[float] = None


class PurchaseOrder(BaseModel):
    po_id: str
    vendor_name: Optional[str] = None
    po_date: Optional[str] = None
    po_lines: List[POLine]


# ── Scoring & Verification Schemas ────────────────────────────────────────────

class CandidateMatch(BaseModel):
    po_line_no: int
    description: str
    ordered_qty: float
    uom: str
    unit_price: float
    lexical_score: float
    semantic_score: float
    hybrid_score: float
    is_semantic_routed: bool


class VerificationFlags(BaseModel):
    has_discrepancy: bool = False
    qty_flag: str = "QTY_MATCH"          # QTY_MATCH, FLAG_QTY_OVERBILLING, FLAG_QTY_UNDERBILLING
    price_flag: str = "PRICE_MATCH"      # PRICE_MATCH, FLAG_PRICE_VARIANCE
    math_flag: str = "MATH_CORRECT"      # MATH_CORRECT, FLAG_MATH_ERROR
    uom_flag: str = "UOM_MATCH"          # UOM_MATCH, FLAG_UOM_MISMATCH
    price_diff_pct: float = 0.0
    qty_diff: float = 0.0
    math_diff: float = 0.0
    discrepancy_details: List[str] = []


class LineReconciliationResult(BaseModel):
    line_no: int
    invoice_description: str
    invoice_qty: float
    invoice_uom: str
    invoice_unit_price: float
    invoice_line_total: float
    
    status: ReconciliationStatus
    assigned_po_line_no: Optional[int] = None
    assigned_po_description: Optional[str] = None
    assigned_po_qty: Optional[float] = None
    assigned_po_uom: Optional[str] = None
    assigned_po_unit_price: Optional[float] = None
    
    score: float = 0.0
    confidence_margin: float = 0.0
    is_semantic_routed: bool = False
    
    verification: VerificationFlags = Field(default_factory=VerificationFlags)
    top_candidates: List[CandidateMatch] = []
    
    is_reviewed: bool = False
    review_action: Optional[str] = None
    review_notes: Optional[str] = None


class ReconciliationSummary(BaseModel):
    total_invoice_lines: int
    total_po_lines: int
    confidently_matched_count: int
    confidently_matched_pct: float
    ambiguous_count: int
    ambiguous_pct: float
    unmatched_count: int
    unmatched_pct: float
    discrepancies_count: int
    first_pass_rate_pct: float
    overall_status: str  # CLEAN_AUTO_ACCEPT, REQUIRES_HUMAN_REVIEW, ACTION_REQUIRED


class AuditLogEntry(BaseModel):
    timestamp: str
    action: str
    line_no: Optional[int] = None
    details: str
    user: str = "AP Reviewer"


class ReconciliationResponse(BaseModel):
    invoice_id: str
    po_id: Optional[str] = None
    vendor_name: Optional[str] = None
    scenario_category: Optional[str] = "custom"
    summary: ReconciliationSummary
    lines: List[LineReconciliationResult]
    audit_log: List[AuditLogEntry] = []


class ReviewActionRequest(BaseModel):
    line_no: int
    action: ReviewActionType
    override_po_line_no: Optional[int] = None
    notes: Optional[str] = ""
