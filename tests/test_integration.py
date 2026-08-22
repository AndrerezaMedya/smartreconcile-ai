"""
End-to-End Integration Tests for Starlette Reconciliation Service.
Uses Starlette TestClient to test all REST API routes and human review workflows.
"""

import unittest
from starlette.testclient import TestClient
from app.main import app
from app.core.models import ReviewActionType


class TestAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

    def test_index_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("SmartReconcile", res.text)

    def test_list_scenarios(self):
        res = self.client.get("/api/scenarios")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_scenarios", data)
        self.assertGreaterEqual(data["total_scenarios"], 20)

    def test_get_single_scenario(self):
        res = self.client.get("/api/scenarios/INV-2026-001")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["invoice"]["invoice_id"], "INV-2026-001")
        self.assertEqual(data["purchase_order"]["po_id"], "PO-2026-001")

    def test_reconcile_clean_invoice_endpoint(self):
        scenario_res = self.client.get("/api/scenarios/INV-2026-001")
        payload = scenario_res.json()

        res = self.client.post("/api/reconcile", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["invoice_id"], "INV-2026-001")
        self.assertEqual(data["summary"]["overall_status"], "CLEAN_AUTO_ACCEPT")
        self.assertEqual(data["summary"]["first_pass_rate_pct"], 100.0)

    def test_reconcile_and_human_review_workflow(self):
        # Scenario INV-2026-020 contains an unmatched freight fee
        scenario_res = self.client.get("/api/scenarios/INV-2026-020")
        payload = scenario_res.json()

        rec_res = self.client.post("/api/reconcile", json=payload)
        self.assertEqual(rec_res.status_code, 200)
        rec_data = rec_res.json()
        
        # Verify line 4 (Freight fee) is UNMATCHED
        line_4 = next(l for l in rec_data["lines"] if l["line_no"] == 4)
        self.assertEqual(line_4["status"], "UNMATCHED")

        # Submit human review action: Reject line 4
        review_payload = {
            "line_no": 4,
            "action": ReviewActionType.REJECT.value,
            "notes": "Freight fee rejected — FOB Destination terms agreed in contract"
        }
        rev_res = self.client.post("/api/review", json=review_payload)
        self.assertEqual(rev_res.status_code, 200)
        rev_data = rev_res.json()
        
        updated_line_4 = next(l for l in rev_data["lines"] if l["line_no"] == 4)
        self.assertTrue(updated_line_4["is_reviewed"])
        self.assertEqual(updated_line_4["review_action"], "REJECTED")

        # Verify the rejection is recorded in this session's audit trail.
        # (No GET /api/audit-log — a browse-all compliance log endpoint is out
        # of scope for the qualifying MVP; the response's own audit_log already
        # carries every event for this reconciliation session.)
        self.assertTrue(any(log["action"] == "HUMAN_REJECTION" for log in rev_data["audit_log"]))

    def test_batch_simulation_benchmark_endpoint(self):
        res = self.client.post("/api/benchmark/run-all")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_invoices_tested"], 24)
        self.assertEqual(data["total_invoice_lines"], 106)
        self.assertEqual(data["first_pass_match_rate_pct"], 75.5)


if __name__ == "__main__":
    unittest.main()
