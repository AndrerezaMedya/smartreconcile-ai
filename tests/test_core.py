"""
Unit tests for Core Reconciliation Services using built-in unittest:
  - HybridMatcher
  - NumericVerifier
  - ReconciliationEngine
"""

import unittest
from app.core.models import (
    Invoice,
    InvoiceLine,
    PurchaseOrder,
    POLine,
    ReconciliationStatus,
    ReviewActionRequest,
    ReviewActionType
)
from app.core.matcher import HybridMatcher
from app.core.verifier import NumericVerifier
from app.core.reconciler import ReconciliationEngine


class TestCoreReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = HybridMatcher()
        cls.verifier = NumericVerifier()
        cls.engine = ReconciliationEngine(matcher=cls.matcher)

    # ── 1. MATCHER TESTS ──────────────────────────────────────────────────────

    def test_jaccard_similarity(self):
        # Exact match
        self.assertEqual(HybridMatcher.jaccard("Pipa Galvanis 2 inch", "Pipa Galvanis 2 inch"), 1.0)
        # Disjoint strings
        self.assertEqual(HybridMatcher.jaccard("Besi Beton D12", "Oli Gardan SAE 90"), 0.0)
        # Partial overlap
        score = HybridMatcher.jaccard("Pipa Galvanis 2 inch Sch 40", "Galvanized Steel Pipe 2 inch Sch 40")
        self.assertTrue(0.0 < score < 1.0)

    def test_greedy_1_to_1_mutual_exclusivity(self):
        """Ensure two invoice lines competing for same PO line are assigned mutually exclusively."""
        inv_lines = [
            InvoiceLine(line_no=1, description="Deep Groove Ball Bearing 6205 2RS SKF", qty=10, uom="pcs", unit_price=65000, line_total=650000),
            InvoiceLine(line_no=2, description="Deep Groove Ball Bearing 6205 ZZ SKF", qty=10, uom="pcs", unit_price=62000, line_total=620000),
        ]
        po_lines = [
            POLine(po_line_no=1, description="DEEP GROOVE BALL BEARING 6205 2RS RUBBER SEAL SKF", ordered_qty=10, uom="pcs", unit_price=65000),
            POLine(po_line_no=2, description="DEEP GROOVE BALL BEARING 6205 ZZ METAL SHIELD SKF", ordered_qty=10, uom="pcs", unit_price=62000),
        ]
        results = self.matcher.match_and_assign(inv_lines, po_lines)
        
        self.assertEqual(len(results), 2)
        assigned_po_nos = [r["assigned_po"].po_line_no for r in results if r["assigned_po"]]
        self.assertEqual(len(set(assigned_po_nos)), 2)  # Mutual exclusivity holds
        self.assertEqual(results[0]["assigned_po"].po_line_no, 1)
        self.assertEqual(results[1]["assigned_po"].po_line_no, 2)

    # ── 2. VERIFIER TESTS ─────────────────────────────────────────────────────

    def test_verifier_clean_match(self):
        inv_line = InvoiceLine(line_no=1, description="MCB 3P 16A", qty=10, uom="pcs", unit_price=185000, line_total=1850000)
        po_line = POLine(po_line_no=1, description="MCB 3P 16A", ordered_qty=10, uom="pcs", unit_price=185000)
        
        flags = self.verifier.verify(inv_line, po_line)
        self.assertFalse(flags.has_discrepancy)
        self.assertEqual(flags.qty_flag, "QTY_MATCH")
        self.assertEqual(flags.price_flag, "PRICE_MATCH")
        self.assertEqual(flags.math_flag, "MATH_CORRECT")
        self.assertEqual(flags.uom_flag, "UOM_MATCH")

    def test_verifier_price_variance(self):
        # Invoiced price 200,000 vs PO price 180,000 (> 1% tolerance)
        inv_line = InvoiceLine(line_no=1, description="MCB 3P 16A", qty=10, uom="pcs", unit_price=200000, line_total=2000000)
        po_line = POLine(po_line_no=1, description="MCB 3P 16A", ordered_qty=10, uom="pcs", unit_price=180000)
        
        flags = self.verifier.verify(inv_line, po_line)
        self.assertTrue(flags.has_discrepancy)
        self.assertEqual(flags.price_flag, "FLAG_PRICE_VARIANCE")

    def test_verifier_quantity_overbilling(self):
        # Invoiced qty 15 vs PO ordered 10
        inv_line = InvoiceLine(line_no=1, description="MCB 3P 16A", qty=15, uom="pcs", unit_price=185000, line_total=2775000)
        po_line = POLine(po_line_no=1, description="MCB 3P 16A", ordered_qty=10, uom="pcs", unit_price=185000)
        
        flags = self.verifier.verify(inv_line, po_line)
        self.assertTrue(flags.has_discrepancy)
        self.assertEqual(flags.qty_flag, "FLAG_QTY_OVERBILLING")

    def test_verifier_math_arithmetic_error(self):
        # 10 * 50000 = 500,000, but line_total is written as 550,000
        inv_line = InvoiceLine(line_no=1, description="MCB 3P 16A", qty=10, uom="pcs", unit_price=50000, line_total=550000)
        po_line = POLine(po_line_no=1, description="MCB 3P 16A", ordered_qty=10, uom="pcs", unit_price=50000)
        
        flags = self.verifier.verify(inv_line, po_line)
        self.assertTrue(flags.has_discrepancy)
        self.assertEqual(flags.math_flag, "FLAG_MATH_ERROR")

    def test_verifier_uom_incompatibility(self):
        # Invoiced in 'roll' vs PO in 'meter'
        inv_line = InvoiceLine(line_no=1, description="Kabel NYY 4x2.5", qty=5, uom="roll", unit_price=2500000, line_total=12500000)
        po_line = POLine(po_line_no=1, description="Kabel NYY 4x2.5", ordered_qty=500, uom="meter", unit_price=25000)
        
        flags = self.verifier.verify(inv_line, po_line)
        self.assertTrue(flags.has_discrepancy)
        self.assertEqual(flags.uom_flag, "FLAG_UOM_MISMATCH")

    # ── 3. RECONCILIATION ENGINE TESTS ────────────────────────────────────────

    def test_engine_clean_auto_match(self):
        inv = Invoice(
            invoice_id="INV-CLEAN-001",
            po_id="PO-CLEAN-001",
            vendor_name="PT Test Industrial",
            invoice_lines=[
                InvoiceLine(line_no=1, description="GALVANIZED STEEL PIPE 2 INCH SCH 40 L=6M", qty=20, uom="batang", unit_price=450000, line_total=9000000),
                InvoiceLine(line_no=2, description="BRASS BALL VALVE 1 INCH PN16 THREADED", qty=8, uom="pcs", unit_price=175000, line_total=1400000),
            ]
        )
        po = PurchaseOrder(
            po_id="PO-CLEAN-001",
            vendor_name="PT Test Industrial",
            po_lines=[
                POLine(po_line_no=1, description="GALVANIZED STEEL PIPE 2 INCH SCH 40 L=6M", ordered_qty=20, uom="batang", unit_price=450000),
                POLine(po_line_no=2, description="BRASS BALL VALVE 1 INCH PN16 THREADED", ordered_qty=8, uom="pcs", unit_price=175000),
            ]
        )
        res = self.engine.reconcile(inv, po)
        self.assertEqual(res.summary.overall_status, "CLEAN_AUTO_ACCEPT")
        self.assertEqual(res.lines[0].status, ReconciliationStatus.MATCHED)
        self.assertEqual(res.lines[0].assigned_po_line_no, 1)
        self.assertEqual(res.lines[1].status, ReconciliationStatus.MATCHED)
        self.assertEqual(res.lines[1].assigned_po_line_no, 2)

    def test_engine_reconcile_and_review_workflow(self):
        inv = Invoice(
            invoice_id="INV-TEST-001",
            po_id="PO-TEST-001",
            vendor_name="PT Test Industrial",
            invoice_lines=[
                InvoiceLine(line_no=1, description="Pipa Galv 2 inch Sch 40", qty=20, uom="batang", unit_price=450000, line_total=9000000),
                InvoiceLine(line_no=2, description="Biaya Ekspedisi Pengiriman", qty=1, uom="trip", unit_price=500000, line_total=500000), # Unmatched
            ]
        )
        po = PurchaseOrder(
            po_id="PO-TEST-001",
            vendor_name="PT Test Industrial",
            po_lines=[
                POLine(po_line_no=1, description="GALVANIZED STEEL PIPE 2 INCH SCH 40 L=6M", ordered_qty=20, uom="batang", unit_price=450000),
                POLine(po_line_no=2, description="CARBON STEEL PIPE 2 INCH SCH 40 L=6M", ordered_qty=10, uom="batang", unit_price=380000),
            ]
        )
        
        res = self.engine.reconcile(inv, po)
        self.assertEqual(res.invoice_id, "INV-TEST-001")
        self.assertEqual(len(res.lines), 2)
        self.assertEqual(res.lines[0].assigned_po_line_no, 1)
        self.assertEqual(res.lines[1].status, ReconciliationStatus.UNMATCHED)
        
        # Test Human Review Action: Approve line 1
        req_approve = ReviewActionRequest(
            line_no=1,
            action=ReviewActionType.APPROVE,
            notes="Reviewer confirmed correct match with Galvanized pipe"
        )
        res_approved = self.engine.apply_review_action(req_approve)
        self.assertEqual(res_approved.lines[0].status, ReconciliationStatus.MATCHED)
        self.assertTrue(res_approved.lines[0].is_reviewed)

        # Test Human Review Action: Override line 1 to PO line 2
        req_override = ReviewActionRequest(
            line_no=1,
            action=ReviewActionType.OVERRIDE,
            override_po_line_no=2,
            notes="Reviewer manually re-mapped to carbon steel"
        )
        updated_res = self.engine.apply_review_action(req_override)
        self.assertEqual(updated_res.lines[0].assigned_po_line_no, 2)
        self.assertTrue(updated_res.lines[0].is_reviewed)
        self.assertTrue(len(updated_res.audit_log) >= 3)


if __name__ == "__main__":
    unittest.main()
