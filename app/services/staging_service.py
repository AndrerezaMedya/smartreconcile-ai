"""
Staging & Reconciliation Orchestrator for SmartReconcile AI.
Connects Validation -> Extraction -> PO Lookup -> AI Reconciliation -> Database Staging -> Human Review -> Permanent Commit.
"""

from typing import Dict, Any, Optional, List
from app.core.models import (
    Invoice,
    PurchaseOrder,
    POLine,
    ReconciliationResponse,
    ReviewActionRequest,
    ReviewActionType,
    ReconciliationStatus,
    VerificationFlags
)
from app.core.reconciler import ReconciliationEngine
from app.core.matcher import HybridMatcher
from app.db.po_repository import PORepository
from app.db.staging_repository import StagingRepository
from app.services.validator import FileValidator
from app.services.extractor import InvoiceExtractor

# Singleton Matcher and Engine instance
_matcher = HybridMatcher()
_engine = ReconciliationEngine(matcher=_matcher)


class StagingService:

    @classmethod
    def process_and_stage_file(
        cls,
        filename: str,
        content_bytes: bytes,
        po_id_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete ingestion pipeline:
        1. Validate file
        2. Extract invoice header & lines
        3. Resolve PO from database
        4. Run AI matching & 4-way verification (FROZEN ENGINE)
        5. Persist staged draft to SQLite database
        """
        # Step 1: Validate file
        is_valid, err_msg, ext = FileValidator.validate(filename, content_bytes)
        if not is_valid:
            raise ValueError(f"File validation failed: {err_msg}")

        # Step 2: Extract Invoice data
        invoice, extracted_po_ref = InvoiceExtractor.extract(filename, content_bytes)
        if not invoice.invoice_lines:
            raise ValueError("No line items could be extracted from the uploaded document.")

        target_po_id = po_id_override or extracted_po_ref or invoice.po_id

        # Step 3: Resolve PO from database
        po = PORepository.get_po_by_id(target_po_id)
        if not po:
            # Fallback: create placeholder PO with empty lines if unknown PO
            po = PurchaseOrder(
                po_id=target_po_id or "PO-UNRESOLVED",
                vendor_name=invoice.vendor_name,
                po_lines=[]
            )

        # Step 4: Run AI matching and 4-way deterministic verification (FROZEN CORE)
        rec_response = _engine.reconcile(invoice, po)

        # Step 5: Save staged draft to SQLite database
        staging_id = StagingRepository.create_staged_reconciliation(rec_response, file_name=filename)

        staged_data = StagingRepository.get_staged_reconciliation(staging_id)
        return staged_data

    @classmethod
    def apply_staged_review_action(
        cls,
        staging_id: str,
        line_no: int,
        action: ReviewActionType,
        override_po_line_no: Optional[int] = None,
        notes: str = "",
        manual_edits: Optional[Dict[str, Any]] = None,
        reviewer_name: str = "AP Reviewer"
    ) -> Dict[str, Any]:
        """
        Applies a reviewer action (Approve / Override / Reject / Manual Edit) to a staged draft line.
        """
        staged = StagingRepository.get_staged_reconciliation(staging_id)
        if not staged:
            raise ValueError(f"Staging ID '{staging_id}' not found.")

        target_line = next((l for l in staged["lines"] if l.line_no == line_no), None)
        if not target_line:
            raise ValueError(f"Line #{line_no} not found in Staging ID '{staging_id}'.")

        po = PORepository.get_po_by_id(staged["po_id"])
        po_lines = po.po_lines if po else []

        action_name = "HUMAN_APPROVAL"
        if action == ReviewActionType.APPROVE:
            target_line.status = ReconciliationStatus.MATCHED
            target_line.is_reviewed = True
            target_line.review_action = "APPROVED"
            target_line.review_notes = notes or "Approved by reviewer"
            action_name = "HUMAN_APPROVAL"

        elif action == ReviewActionType.OVERRIDE:
            if override_po_line_no is None:
                raise ValueError("override_po_line_no is required for OVERRIDE action.")

            chosen_po = next((pl for pl in po_lines if pl.po_line_no == override_po_line_no), None)
            if not chosen_po:
                raise ValueError(f"PO Line #{override_po_line_no} not found in PO {staged['po_id']}.")

            target_line.assigned_po_line_no = chosen_po.po_line_no
            target_line.assigned_po_description = chosen_po.description
            target_line.assigned_po_qty = chosen_po.ordered_qty
            target_line.assigned_po_uom = chosen_po.uom
            target_line.assigned_po_unit_price = chosen_po.unit_price
            target_line.status = ReconciliationStatus.MATCHED
            target_line.is_reviewed = True
            target_line.review_action = f"OVERRIDDEN_TO_PO_LINE_{chosen_po.po_line_no}"
            target_line.review_notes = notes or f"Manual override to PO line #{chosen_po.po_line_no}"

            # Re-run 4-way verification with overridden PO line
            target_line.verification = _engine.verifier.verify(
                target_line.to_invoice_line(),
                chosen_po
            )
            action_name = "HUMAN_OVERRIDE"

        elif action == ReviewActionType.REJECT:
            target_line.status = ReconciliationStatus.UNMATCHED
            target_line.is_reviewed = True
            target_line.review_action = "REJECTED"
            target_line.review_notes = notes or "Line item rejected by reviewer"
            action_name = "HUMAN_REJECTION"

        # Apply any manual attribute edits if provided
        if manual_edits:
            if "invoice_qty" in manual_edits:
                target_line.invoice_qty = float(manual_edits["invoice_qty"])
            if "invoice_unit_price" in manual_edits:
                target_line.invoice_unit_price = float(manual_edits["invoice_unit_price"])
            if "invoice_line_total" in manual_edits:
                target_line.invoice_line_total = float(manual_edits["invoice_line_total"])
            else:
                target_line.invoice_line_total = target_line.invoice_qty * target_line.invoice_unit_price

            assigned_po = next((pl for pl in po_lines if pl.po_line_no == target_line.assigned_po_line_no), None) if target_line.assigned_po_line_no else None
            target_line.verification = _engine.verifier.verify(
                target_line.to_invoice_line(),
                assigned_po
            )
            action_name = "HUMAN_EDIT"

        # Update in database
        StagingRepository.update_staged_line(
            staging_id=staging_id,
            line_no=line_no,
            line_result=target_line,
            action_name=action_name,
            user_name=reviewer_name
        )

        return StagingRepository.get_staged_reconciliation(staging_id)

    @classmethod
    def commit_staged_draft(
        cls,
        staging_id: str,
        reviewer_name: str = "AP Lead",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Commits a staged draft into the permanent committed_invoices ledger.
        """
        return StagingRepository.commit_reconciliation(
            staging_id=staging_id,
            reviewer_name=reviewer_name,
            notes=notes
        )
