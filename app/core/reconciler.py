"""
Master Reconciliation Engine and State Manager.
Coordinates matching, verification, confidence gating, and human review overrides.
"""

from datetime import datetime
from typing import Dict, List, Optional
from app.core.config import CONFIDENCE_MARGIN_THRESHOLD
from app.core.models import (
    Invoice,
    PurchaseOrder,
    LineReconciliationResult,
    ReconciliationStatus,
    ReconciliationSummary,
    ReconciliationResponse,
    AuditLogEntry,
    ReviewActionRequest,
    ReviewActionType
)
from app.core.matcher import HybridMatcher
from app.core.verifier import NumericVerifier


class ReconciliationEngine:
    """
    Stateful reconciliation manager for an active invoice-to-PO session.
    """

    def __init__(self, matcher: Optional[HybridMatcher] = None):
        self.matcher = matcher or HybridMatcher()
        self.verifier = NumericVerifier()
        self.current_invoice: Optional[Invoice] = None
        self.current_po: Optional[PurchaseOrder] = None
        self.current_results: List[LineReconciliationResult] = []
        self.audit_log: List[AuditLogEntry] = []

    def reconcile(
        self,
        invoice: Invoice,
        po: PurchaseOrder
    ) -> ReconciliationResponse:
        """
        Executes end-to-end reconciliation on the provided invoice and PO.
        """
        self.current_invoice = invoice
        self.current_po = po
        self.audit_log = []

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.audit_log.append(AuditLogEntry(
            timestamp=now_str,
            action="INITIAL_RECONCILIATION",
            details=f"Reconciled Invoice {invoice.invoice_id} ({len(invoice.invoice_lines)} lines) against PO {po.po_id} ({len(po.po_lines)} lines)."
        ))

        # 1. Match & Assign
        match_outputs = self.matcher.match_and_assign(invoice.invoice_lines, po.po_lines)

        # 2. Verify & Classify
        results: List[LineReconciliationResult] = []
        for mo in match_outputs:
            il = mo["invoice_line"]
            assigned_po = mo["assigned_po"]
            score = mo["score"]
            margin = mo["confidence_margin"]
            is_semantic_routed = mo["is_semantic_routed"]
            top_candidates = mo["top_candidates"]

            # 4-way verification
            v_flags = self.verifier.verify(il, assigned_po)

            # Confidence Gating Classification
            if assigned_po is None:
                status = ReconciliationStatus.UNMATCHED
            elif margin >= CONFIDENCE_MARGIN_THRESHOLD and not v_flags.has_discrepancy:
                status = ReconciliationStatus.MATCHED
            else:
                status = ReconciliationStatus.AMBIGUOUS

            results.append(LineReconciliationResult(
                line_no=il.line_no,
                invoice_description=il.description,
                invoice_qty=il.qty,
                invoice_uom=il.uom,
                invoice_unit_price=il.unit_price,
                invoice_line_total=il.line_total,
                status=status,
                assigned_po_line_no=assigned_po.po_line_no if assigned_po else None,
                assigned_po_description=assigned_po.description if assigned_po else None,
                assigned_po_qty=assigned_po.ordered_qty if assigned_po else None,
                assigned_po_uom=assigned_po.uom if assigned_po else None,
                assigned_po_unit_price=assigned_po.unit_price if assigned_po else None,
                score=score,
                confidence_margin=margin,
                is_semantic_routed=is_semantic_routed,
                verification=v_flags,
                top_candidates=top_candidates,
                is_reviewed=False
            ))

        self.current_results = results
        summary = self._calculate_summary(results, len(po.po_lines))

        return ReconciliationResponse(
            invoice_id=invoice.invoice_id,
            po_id=po.po_id,
            vendor_name=invoice.vendor_name,
            scenario_category=invoice.scenario_category,
            summary=summary,
            lines=results,
            audit_log=self.audit_log
        )

    def apply_review_action(
        self,
        request: ReviewActionRequest,
        reviewer_name: str = "AP Specialist"
    ) -> ReconciliationResponse:
        """
        Applies a human reviewer decision (Approve / Override / Reject) to a line.
        """
        if not self.current_invoice or not self.current_po:
            raise ValueError("No active reconciliation session found.")

        target_line = next((r for r in self.current_results if r.line_no == request.line_no), None)
        if not target_line:
            raise ValueError(f"Line number {request.line_no} not found.")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inv_line = next(il for il in self.current_invoice.invoice_lines if il.line_no == request.line_no)

        if request.action == ReviewActionType.APPROVE:
            target_line.is_reviewed = True
            target_line.review_action = "APPROVED"
            target_line.review_notes = request.notes
            target_line.status = ReconciliationStatus.MATCHED
            self.audit_log.append(AuditLogEntry(
                timestamp=now_str,
                action="HUMAN_APPROVAL",
                line_no=request.line_no,
                details=f"Reviewer approved match with PO Line #{target_line.assigned_po_line_no}. Notes: {request.notes or 'None'}",
                user=reviewer_name
            ))

        elif request.action == ReviewActionType.OVERRIDE:
            if request.override_po_line_no is None:
                raise ValueError("override_po_line_no must be provided for OVERRIDE action.")

            new_po = next((pl for pl in self.current_po.po_lines if pl.po_line_no == request.override_po_line_no), None)
            if not new_po:
                raise ValueError(f"PO Line #{request.override_po_line_no} does not exist.")

            # Reassign and re-verify
            target_line.assigned_po_line_no = new_po.po_line_no
            target_line.assigned_po_description = new_po.description
            target_line.assigned_po_qty = new_po.ordered_qty
            target_line.assigned_po_uom = new_po.uom
            target_line.assigned_po_unit_price = new_po.unit_price
            target_line.verification = self.verifier.verify(inv_line, new_po)
            target_line.is_reviewed = True
            target_line.review_action = "OVERRIDDEN"
            target_line.review_notes = request.notes
            target_line.status = ReconciliationStatus.MATCHED

            self.audit_log.append(AuditLogEntry(
                timestamp=now_str,
                action="HUMAN_OVERRIDE",
                line_no=request.line_no,
                details=f"Reviewer manually re-assigned line to PO Line #{new_po.po_line_no} ('{new_po.description}'). Notes: {request.notes or 'None'}",
                user=reviewer_name
            ))

        elif request.action == ReviewActionType.REJECT:
            target_line.assigned_po_line_no = None
            target_line.assigned_po_description = None
            target_line.assigned_po_qty = None
            target_line.assigned_po_uom = None
            target_line.assigned_po_unit_price = None
            target_line.status = ReconciliationStatus.UNMATCHED
            target_line.is_reviewed = True
            target_line.review_action = "REJECTED"
            target_line.review_notes = request.notes
            target_line.verification.has_discrepancy = True
            target_line.verification.discrepancy_details.append(f"Rejected by human reviewer: {request.notes or 'No PO authorization'}")

            self.audit_log.append(AuditLogEntry(
                timestamp=now_str,
                action="HUMAN_REJECTION",
                line_no=request.line_no,
                details=f"Reviewer rejected line item. Exception flagged. Notes: {request.notes or 'None'}",
                user=reviewer_name
            ))

        summary = self._calculate_summary(self.current_results, len(self.current_po.po_lines))
        return ReconciliationResponse(
            invoice_id=self.current_invoice.invoice_id,
            po_id=self.current_po.po_id,
            vendor_name=self.current_invoice.vendor_name,
            scenario_category=self.current_invoice.scenario_category,
            summary=summary,
            lines=self.current_results,
            audit_log=self.audit_log
        )

    def _calculate_summary(
        self,
        results: List[LineReconciliationResult],
        total_po_lines: int
    ) -> ReconciliationSummary:
        n_total = len(results)
        if n_total == 0:
            return ReconciliationSummary(
                total_invoice_lines=0,
                total_po_lines=total_po_lines,
                confidently_matched_count=0,
                confidently_matched_pct=0.0,
                ambiguous_count=0,
                ambiguous_pct=0.0,
                unmatched_count=0,
                unmatched_pct=0.0,
                discrepancies_count=0,
                first_pass_rate_pct=0.0,
                overall_status="NO_DATA"
            )

        n_matched = sum(1 for r in results if r.status == ReconciliationStatus.MATCHED)
        n_ambiguous = sum(1 for r in results if r.status == ReconciliationStatus.AMBIGUOUS)
        n_unmatched = sum(1 for r in results if r.status == ReconciliationStatus.UNMATCHED)
        n_discrepancies = sum(1 for r in results if r.verification.has_discrepancy)

        matched_pct = round((n_matched / n_total) * 100, 1)
        ambiguous_pct = round((n_ambiguous / n_total) * 100, 1)
        unmatched_pct = round((n_unmatched / n_total) * 100, 1)
        first_pass_pct = matched_pct

        if n_ambiguous == 0 and n_unmatched == 0 and n_discrepancies == 0:
            overall_status = "CLEAN_AUTO_ACCEPT"
        elif n_unmatched > 0 or n_discrepancies > 0:
            overall_status = "ACTION_REQUIRED"
        else:
            overall_status = "REQUIRES_HUMAN_REVIEW"

        return ReconciliationSummary(
            total_invoice_lines=n_total,
            total_po_lines=total_po_lines,
            confidently_matched_count=n_matched,
            confidently_matched_pct=matched_pct,
            ambiguous_count=n_ambiguous,
            ambiguous_pct=ambiguous_pct,
            unmatched_count=n_unmatched,
            unmatched_pct=unmatched_pct,
            discrepancies_count=n_discrepancies,
            first_pass_rate_pct=first_pass_pct,
            overall_status=overall_status
        )
