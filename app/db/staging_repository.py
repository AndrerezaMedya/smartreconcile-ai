"""
Staging Repository for persisting draft reconciliations and recording committed ledger entries.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from app.core.models import (
    LineReconciliationResult,
    ReconciliationResponse,
    ReconciliationSummary,
    ReconciliationStatus,
    VerificationFlags,
    CandidateMatch,
    AuditLogEntry
)
from app.db.database import get_db_connection


class StagingRepository:

    @staticmethod
    def create_staged_reconciliation(
        rec_response: ReconciliationResponse,
        file_name: Optional[str] = None
    ) -> str:
        """
        Persists an initial reconciliation result into the staged_reconciliations table.
        Returns the unique staging_id.
        """
        staging_id = f"STG-{uuid.uuid4().hex[:8].upper()}"
        conn = get_db_connection()
        cursor = conn.cursor()

        summary = rec_response.summary
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        staged_status = "STAGED_PENDING_REVIEW"

        cursor.execute("""
            INSERT INTO staged_reconciliations (
                staging_id, invoice_id, po_id, vendor_name, file_name, status,
                first_pass_rate, confidently_matched_count, ambiguous_count,
                unmatched_count, discrepancies_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            staging_id, rec_response.invoice_id, rec_response.po_id, rec_response.vendor_name,
            file_name, staged_status, summary.first_pass_rate_pct, summary.confidently_matched_count,
            summary.ambiguous_count, summary.unmatched_count, summary.discrepancies_count,
            now_str, now_str
        ))

        for line in rec_response.lines:
            v = line.verification
            discrepancy_details_json = json.dumps(v.discrepancy_details) if v else "[]"
            top_candidates_json = json.dumps([c.model_dump() for c in line.top_candidates]) if line.top_candidates else "[]"

            cursor.execute("""
                INSERT INTO staged_lines (
                    staging_id, line_no, invoice_description, invoice_qty, invoice_uom,
                    invoice_unit_price, invoice_line_total, status,
                    assigned_po_line_no, assigned_po_description, assigned_po_qty,
                    assigned_po_uom, assigned_po_unit_price, score, confidence_margin,
                    is_semantic_routed, has_discrepancy, qty_flag, price_flag, math_flag,
                    uom_flag, discrepancy_details, top_candidates_json, is_reviewed,
                    review_action, review_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                staging_id, line.line_no, line.invoice_description, line.invoice_qty, line.invoice_uom,
                line.invoice_unit_price, line.invoice_line_total, line.status.value,
                line.assigned_po_line_no, line.assigned_po_description, line.assigned_po_qty,
                line.assigned_po_uom, line.assigned_po_unit_price, line.score, line.confidence_margin,
                int(line.is_semantic_routed), int(v.has_discrepancy) if v else 0,
                v.qty_flag if v else "QTY_MATCH", v.price_flag if v else "PRICE_MATCH",
                v.math_flag if v else "MATH_CORRECT", v.uom_flag if v else "UOM_MATCH",
                discrepancy_details_json, top_candidates_json, int(line.is_reviewed),
                line.review_action, line.review_notes
            ))

        # Initial audit log
        cursor.execute("""
            INSERT INTO audit_logs (staging_id, timestamp, action, line_no, details, user)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            staging_id, now_str, "STAGING_CREATED", None,
            f"Staged draft created for Invoice {rec_response.invoice_id} ({len(rec_response.lines)} lines). Status: {staged_status}.",
            "System Engine"
        ))

        conn.commit()
        conn.close()
        return staging_id

    @staticmethod
    def upsert_staged_reconciliation(
        rec_response: "ReconciliationResponse",
        file_name: Optional[str] = None
    ) -> str:
        """
        Idempotent version of create_staged_reconciliation.
        If a STAGED_PENDING_REVIEW record already exists for the same invoice_id,
        it is deleted first (cascading to staged_lines; audit_logs cleaned manually).
        Records with status COMMITTED are never touched.
        Returns the new staging_id.
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        # Find any existing non-committed staging for this invoice
        cursor.execute(
            "SELECT staging_id FROM staged_reconciliations "
            "WHERE invoice_id = ? AND status = 'STAGED_PENDING_REVIEW'",
            (rec_response.invoice_id,)
        )
        existing_rows = cursor.fetchall()
        for row in existing_rows:
            old_id = row["staging_id"]
            # Delete audit_logs first (no CASCADE defined)
            cursor.execute("DELETE FROM audit_logs WHERE staging_id = ?", (old_id,))
            # Delete staged_reconciliations (staged_lines cascade automatically)
            cursor.execute("DELETE FROM staged_reconciliations WHERE staging_id = ?", (old_id,))

        conn.commit()
        conn.close()

        # Now create a fresh staging record
        return StagingRepository.create_staged_reconciliation(
            rec_response=rec_response,
            file_name=file_name
        )

    @staticmethod
    def get_staged_reconciliation(staging_id: str) -> Optional[Dict[str, Any]]:
        """Fetches full staged reconciliation data by staging_id."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM staged_reconciliations WHERE staging_id = ?", (staging_id,))
        head = cursor.fetchone()
        if not head:
            conn.close()
            return None

        cursor.execute("SELECT * FROM staged_lines WHERE staging_id = ? ORDER BY line_no ASC", (staging_id,))
        line_rows = cursor.fetchall()

        cursor.execute("SELECT * FROM audit_logs WHERE staging_id = ? ORDER BY id ASC", (staging_id,))
        audit_rows = cursor.fetchall()
        conn.close()

        lines = []
        for lr in line_rows:
            details_list = json.loads(lr["discrepancy_details"]) if lr["discrepancy_details"] else []
            cand_list_raw = json.loads(lr["top_candidates_json"]) if lr["top_candidates_json"] else []
            top_candidates = [CandidateMatch(**c) for c in cand_list_raw]

            verification = VerificationFlags(
                has_discrepancy=bool(lr["has_discrepancy"]),
                qty_flag=lr["qty_flag"] or "QTY_MATCH",
                price_flag=lr["price_flag"] or "PRICE_MATCH",
                math_flag=lr["math_flag"] or "MATH_CORRECT",
                uom_flag=lr["uom_flag"] or "UOM_MATCH",
                discrepancy_details=details_list
            )

            lines.append(LineReconciliationResult(
                line_no=lr["line_no"],
                invoice_description=lr["invoice_description"],
                invoice_qty=lr["invoice_qty"],
                invoice_uom=lr["invoice_uom"],
                invoice_unit_price=lr["invoice_unit_price"],
                invoice_line_total=lr["invoice_line_total"],
                status=ReconciliationStatus(lr["status"]),
                assigned_po_line_no=lr["assigned_po_line_no"],
                assigned_po_description=lr["assigned_po_description"],
                assigned_po_qty=lr["assigned_po_qty"],
                assigned_po_uom=lr["assigned_po_uom"],
                assigned_po_unit_price=lr["assigned_po_unit_price"],
                score=lr["score"] or 0.0,
                confidence_margin=lr["confidence_margin"] or 0.0,
                is_semantic_routed=bool(lr["is_semantic_routed"]),
                verification=verification,
                top_candidates=top_candidates,
                is_reviewed=bool(lr["is_reviewed"]),
                review_action=lr["review_action"],
                review_notes=lr["review_notes"]
            ))

        audit_log = [
            AuditLogEntry(
                timestamp=ar["timestamp"],
                action=ar["action"],
                line_no=ar["line_no"],
                details=ar["details"],
                user=ar["user"]
            )
            for ar in audit_rows
        ]

        summary = ReconciliationSummary(
            total_invoice_lines=len(lines),
            total_po_lines=len(lines),
            confidently_matched_count=head["confidently_matched_count"],
            confidently_matched_pct=round((head["confidently_matched_count"] / len(lines) * 100), 1) if lines else 0.0,
            ambiguous_count=head["ambiguous_count"],
            ambiguous_pct=round((head["ambiguous_count"] / len(lines) * 100), 1) if lines else 0.0,
            unmatched_count=head["unmatched_count"],
            unmatched_pct=round((head["unmatched_count"] / len(lines) * 100), 1) if lines else 0.0,
            discrepancies_count=head["discrepancies_count"],
            first_pass_rate_pct=head["first_pass_rate"],
            overall_status="CLEAN_AUTO_ACCEPT" if (head["discrepancies_count"] == 0 and head["ambiguous_count"] == 0 and head["unmatched_count"] == 0) else "REQUIRES_HUMAN_REVIEW"
        )

        return {
            "staging_id": head["staging_id"],
            "invoice_id": head["invoice_id"],
            "po_id": head["po_id"],
            "vendor_name": head["vendor_name"],
            "file_name": head["file_name"],
            "status": head["status"],
            "summary": summary,
            "lines": lines,
            "audit_log": audit_log,
            "created_at": head["created_at"],
            "updated_at": head["updated_at"]
        }

    @staticmethod
    def list_staged_reconciliations() -> List[Dict[str, Any]]:
        """Lists all active staged reconciliation drafts."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM staged_reconciliations ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "staging_id": r["staging_id"],
                "invoice_id": r["invoice_id"],
                "po_id": r["po_id"],
                "vendor_name": r["vendor_name"],
                "file_name": r["file_name"],
                "status": r["status"],
                "first_pass_rate": r["first_pass_rate"],
                "confidently_matched_count": r["confidently_matched_count"],
                "ambiguous_count": r["ambiguous_count"],
                "unmatched_count": r["unmatched_count"],
                "discrepancies_count": r["discrepancies_count"],
                "created_at": r["created_at"]
            }
            for r in rows
        ]

    @staticmethod
    def update_staged_line(
        staging_id: str,
        line_no: int,
        line_result: LineReconciliationResult,
        action_name: str,
        user_name: str = "AP Reviewer"
    ):
        """Updates a line within a staged draft and adds an audit log entry."""
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        v = line_result.verification
        discrepancy_details_json = json.dumps(v.discrepancy_details) if v else "[]"

        cursor.execute("""
            UPDATE staged_lines SET
                status = ?,
                assigned_po_line_no = ?,
                assigned_po_description = ?,
                assigned_po_qty = ?,
                assigned_po_uom = ?,
                assigned_po_unit_price = ?,
                has_discrepancy = ?,
                qty_flag = ?,
                price_flag = ?,
                math_flag = ?,
                uom_flag = ?,
                discrepancy_details = ?,
                is_reviewed = ?,
                review_action = ?,
                review_notes = ?
            WHERE staging_id = ? AND line_no = ?
        """, (
            line_result.status.value,
            line_result.assigned_po_line_no,
            line_result.assigned_po_description,
            line_result.assigned_po_qty,
            line_result.assigned_po_uom,
            line_result.assigned_po_unit_price,
            int(v.has_discrepancy) if v else 0,
            v.qty_flag if v else "QTY_MATCH",
            v.price_flag if v else "PRICE_MATCH",
            v.math_flag if v else "MATH_CORRECT",
            v.uom_flag if v else "UOM_MATCH",
            discrepancy_details_json,
            1,
            line_result.review_action,
            line_result.review_notes,
            staging_id,
            line_no
        ))

        cursor.execute("""
            INSERT INTO audit_logs (staging_id, timestamp, action, line_no, details, user)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            staging_id, now_str, action_name, line_no,
            f"Reviewer action '{action_name}' applied to line #{line_no}. Notes: {line_result.review_notes or 'None'}",
            user_name
        ))

        cursor.execute("UPDATE staged_reconciliations SET updated_at = ? WHERE staging_id = ?", (now_str, staging_id))
        conn.commit()
        conn.close()

    @staticmethod
    def commit_reconciliation(
        staging_id: str,
        reviewer_name: str = "AP Lead",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Commits a staged draft into the permanent committed_invoices ledger.
        """
        staged = StagingRepository.get_staged_reconciliation(staging_id)
        if not staged:
            raise ValueError(f"Staged draft '{staging_id}' not found.")

        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_id = f"COM-{uuid.uuid4().hex[:8].upper()}"

        # Guard: prevent committing the same invoice_id twice (e.g. demo re-run without DB reset)
        cursor.execute(
            "SELECT commit_id FROM committed_invoices WHERE invoice_id = ?",
            (staged["invoice_id"],)
        )
        existing_commit = cursor.fetchone()
        if existing_commit:
            conn.close()
            raise ValueError(
                f"Invoice '{staged['invoice_id']}' has already been committed "
                f"(Commit ID: {existing_commit['commit_id']}). "
                f"To re-run this demo scenario, reset the committed ledger first."
            )

        total_inv_amt = sum(l.invoice_line_total for l in staged["lines"])
        total_matched_amt = sum(
            (l.assigned_po_qty or 0) * (l.assigned_po_unit_price or 0)
            for l in staged["lines"] if l.assigned_po_line_no is not None
        )

        cursor.execute("""
            INSERT INTO committed_invoices (
                commit_id, staging_id, invoice_id, po_id, vendor_name,
                total_invoiced_amount, total_matched_amount, reviewer_name,
                commit_timestamp, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            commit_id, staging_id, staged["invoice_id"], staged["po_id"], staged["vendor_name"],
            total_inv_amt, total_matched_amt, reviewer_name, now_str, notes
        ))

        cursor.execute("UPDATE staged_reconciliations SET status = 'COMMITTED', updated_at = ? WHERE staging_id = ?", (now_str, staging_id))

        cursor.execute("""
            INSERT INTO audit_logs (staging_id, timestamp, action, line_no, details, user)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            staging_id, now_str, "COMMITTED_TO_DATABASE", None,
            f"Reconciliation committed to permanent ledger under Commit ID {commit_id}. Total matched: Rp {total_matched_amt:,.0f}.",
            reviewer_name
        ))

        conn.commit()
        conn.close()

        return {
            "commit_id": commit_id,
            "staging_id": staging_id,
            "invoice_id": staged["invoice_id"],
            "po_id": staged["po_id"],
            "vendor_name": staged["vendor_name"],
            "total_invoiced_amount": total_inv_amt,
            "total_matched_amount": total_matched_amt,
            "reviewer_name": reviewer_name,
            "commit_timestamp": now_str,
            "status": "COMMITTED"
        }

    @staticmethod
    def list_committed_invoices() -> List[Dict[str, Any]]:
        """Lists all permanently committed reconciliations."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM committed_invoices ORDER BY commit_timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "commit_id": r["commit_id"],
                "staging_id": r["staging_id"],
                "invoice_id": r["invoice_id"],
                "po_id": r["po_id"],
                "vendor_name": r["vendor_name"],
                "total_invoiced_amount": r["total_invoiced_amount"],
                "total_matched_amount": r["total_matched_amount"],
                "reviewer_name": r["reviewer_name"],
                "commit_timestamp": r["commit_timestamp"],
                "notes": r["notes"]
            }
            for r in rows
        ]
