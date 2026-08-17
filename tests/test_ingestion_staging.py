"""
Automated Test Suite for Ingestion, Validation, Staging, Review, and Database Commit.
Tests the complete enterprise product workflow.
"""

import io
import unittest
from pathlib import Path
from starlette.testclient import TestClient

from app.main import app
from app.core.models import ReviewActionType
from app.services.validator import FileValidator
from app.services.extractor import InvoiceExtractor
from app.services.staging_service import StagingService
from app.db.staging_repository import StagingRepository
from app.db.po_repository import PORepository


class TestIngestionAndStagingWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.demo_pdf_path = Path("data/demo_invoices_pdf/INV-2026-001.pdf")
        cls.exception_pdf_path = Path("data/demo_invoices_pdf/INV-2026-020.pdf")

    def test_file_validator(self):
        # Valid PDF
        pdf_bytes = self.demo_pdf_path.read_bytes()
        valid, err, ext = FileValidator.validate("invoice.pdf", pdf_bytes)
        self.assertTrue(valid)
        self.assertEqual(ext, ".pdf")

        # Invalid extension
        valid, err, ext = FileValidator.validate("invoice.exe", b"malicious content")
        self.assertFalse(valid)
        self.assertIn("Unsupported file extension", err)

        # Empty file
        valid, err, ext = FileValidator.validate("empty.pdf", b"")
        self.assertFalse(valid)
        self.assertIn("empty", err)

    def test_pdf_extraction(self):
        pdf_bytes = self.demo_pdf_path.read_bytes()
        inv, po_ref = InvoiceExtractor.extract("INV-2026-001.pdf", pdf_bytes)
        self.assertEqual(inv.invoice_id, "INV-2026-001")
        self.assertEqual(po_ref, "PO-2026-001")
        self.assertEqual(inv.vendor_name, "PT Sumber Baja Perkasa")
        self.assertEqual(len(inv.invoice_lines), 5)

    def test_csv_extraction(self):
        csv_content = (
            "# INVOICE: INV-CSV-999\n"
            "# PO: PO-2026-001\n"
            "# VENDOR: PT Sumber Baja Perkasa\n"
            "Line,Description,Qty,UOM,Unit Price,Total\n"
            "1,Pipa Galvanis 2 inch Sch 40,20,batang,450000,9000000\n"
            "2,Elbow 90 Deg 2 inch Drat Galv,15,pcs,125000,1875000\n"
        ).encode("utf-8")

        inv, po_ref = InvoiceExtractor.extract("invoice.csv", csv_content)
        self.assertEqual(inv.invoice_id, "INV-CSV-999")
        self.assertEqual(po_ref, "PO-2026-001")
        self.assertEqual(len(inv.invoice_lines), 2)
        self.assertEqual(inv.invoice_lines[0].unit_price, 450000.0)

    def test_complete_staging_and_commit_service(self):
        pdf_bytes = self.exception_pdf_path.read_bytes()

        # Step 1: Process and Stage
        staged = StagingService.process_and_stage_file("INV-2026-020.pdf", pdf_bytes)
        staging_id = staged["staging_id"]
        self.assertTrue(staging_id.startswith("STG-"))
        self.assertEqual(staged["status"], "STAGED_PENDING_REVIEW")

        # Step 2: Review line #4 (Freight surcharge)
        updated = StagingService.apply_staged_review_action(
            staging_id=staging_id,
            line_no=4,
            action=ReviewActionType.REJECT,
            notes="Freight fee rejected by reviewer"
        )
        l4 = next(l for l in updated["lines"] if l.line_no == 4)
        self.assertTrue(l4.is_reviewed)
        self.assertEqual(l4.review_action, "REJECTED")

        # Step 3: Commit to SQLite Database
        commit_res = StagingService.commit_staged_draft(
            staging_id=staging_id,
            reviewer_name="Lead Auditor",
            notes="Approved with freight exclusion"
        )
        self.assertTrue(commit_res["commit_id"].startswith("COM-"))
        self.assertEqual(commit_res["status"], "COMMITTED")

    def test_api_upload_and_commit_flow(self):
        pdf_bytes = self.demo_pdf_path.read_bytes()
        files = {"file": ("INV-2026-001.pdf", pdf_bytes, "application/pdf")}

        # 1. POST /api/upload
        res = self.client.post("/api/upload", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        staging_id = data["staging_id"]
        self.assertTrue(staging_id.startswith("STG-"))
        self.assertEqual(data["invoice_id"], "INV-2026-001")

        # 2. GET /api/staging/{id}
        stg_res = self.client.get(f"/api/staging/{staging_id}")
        self.assertEqual(stg_res.status_code, 200)
        self.assertEqual(stg_res.json()["invoice_id"], "INV-2026-001")

        # 3. POST /api/staging/{id}/commit
        commit_payload = {
            "reviewer_name": "Senior AP Manager",
            "notes": "Verified against PO and committed"
        }
        com_res = self.client.post(f"/api/staging/{staging_id}/commit", json=commit_payload)
        self.assertEqual(com_res.status_code, 200)
        com_data = com_res.json()
        self.assertTrue(com_data["commit_id"].startswith("COM-"))

        # 4. GET /api/committed
        ledger_res = self.client.get("/api/committed")
        self.assertEqual(ledger_res.status_code, 200)
        commits = ledger_res.json()["commits"]
        self.assertTrue(any(c["commit_id"] == com_data["commit_id"] for c in commits))


if __name__ == "__main__":
    unittest.main()
