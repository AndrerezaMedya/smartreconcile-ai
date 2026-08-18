"""
REST API routes and request handlers for SmartReconcile AI.
Implements Upload -> Validation -> Extraction -> PO Resolution -> Match -> Stage -> Review -> Commit -> Audit.
"""

import json
from pathlib import Path
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import DATA_DIR
from app.core.models import (
    Invoice,
    PurchaseOrder,
    ReviewActionRequest,
    ReviewActionType
)
from app.core.reconciler import ReconciliationEngine
from app.core.matcher import HybridMatcher
from app.db.po_repository import PORepository
from app.db.staging_repository import StagingRepository
from app.services.staging_service import StagingService

# Singleton engine instance for direct simulation & scenario fallback
_matcher = HybridMatcher()
_engine = ReconciliationEngine(matcher=_matcher)

SIM_INVOICES_PATH = DATA_DIR / "workflow_simulation_invoices.json"


# ── 1. File Ingestion & Staging Endpoints ─────────────────────────────────────

async def upload_invoice_file(request: Request):
    """
    Handles PDF, CSV, or JSON invoice file upload.
    Validates, extracts, resolves PO, reconciles with AI, and creates a staged draft in SQLite.
    """
    try:
        # Check if multipart form upload
        content_type = request.headers.get("content-type", "")
        po_override = request.query_params.get("po_id")

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload_file = form.get("file")
            if not upload_file:
                return JSONResponse({"error": "No file field found in multipart upload."}, status_code=400)
            
            filename = upload_file.filename
            content_bytes = await upload_file.read()
            po_override = form.get("po_id") or po_override
        else:
            # Direct binary or JSON upload with filename in headers
            filename = request.headers.get("x-filename", "uploaded_invoice.pdf")
            content_bytes = await request.body()

        if not content_bytes:
            return JSONResponse({"error": "Uploaded file content is empty."}, status_code=400)

        staged_res = StagingService.process_and_stage_file(
            filename=filename,
            content_bytes=content_bytes,
            po_id_override=po_override
        )

        # Convert to serializable dict
        lines_json = [json.loads(l.model_dump_json()) for l in staged_res["lines"]]
        audit_json = [json.loads(a.model_dump_json()) for a in staged_res["audit_log"]]
        summary_json = json.loads(staged_res["summary"].model_dump_json())

        payload = {
            "staging_id": staged_res["staging_id"],
            "invoice_id": staged_res["invoice_id"],
            "po_id": staged_res["po_id"],
            "vendor_name": staged_res["vendor_name"],
            "file_name": staged_res["file_name"],
            "status": staged_res["status"],
            "summary": summary_json,
            "lines": lines_json,
            "audit_log": audit_json,
            "created_at": staged_res["created_at"]
        }
        return JSONResponse(payload)

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Upload processing error: {str(e)}"}, status_code=500)


async def list_staged_drafts(request: Request):
    """Lists all active staged reconciliation drafts in SQLite."""
    drafts = StagingRepository.list_staged_reconciliations()
    return JSONResponse({"total_drafts": len(drafts), "drafts": drafts})


async def get_staged_draft(request: Request):
    """Fetches a specific staged draft by staging_id."""
    staging_id = request.path_params.get("staging_id")
    staged = StagingRepository.get_staged_reconciliation(staging_id)
    if not staged:
        return JSONResponse({"error": f"Staged draft '{staging_id}' not found."}, status_code=404)

    lines_json = [json.loads(l.model_dump_json()) for l in staged["lines"]]
    audit_json = [json.loads(a.model_dump_json()) for a in staged["audit_log"]]
    summary_json = json.loads(staged["summary"].model_dump_json())

    payload = {
        "staging_id": staged["staging_id"],
        "invoice_id": staged["invoice_id"],
        "po_id": staged["po_id"],
        "vendor_name": staged["vendor_name"],
        "file_name": staged["file_name"],
        "status": staged["status"],
        "summary": summary_json,
        "lines": lines_json,
        "audit_log": audit_json,
        "created_at": staged["created_at"],
        "updated_at": staged["updated_at"]
    }
    return JSONResponse(payload)


async def review_staged_line(request: Request):
    """Applies a human reviewer action (Approve / Override / Reject / Edit) to a staged draft line."""
    try:
        staging_id = request.path_params.get("staging_id")
        data = await request.json()

        line_no = int(data.get("line_no"))
        action_str = data.get("action")
        action = ReviewActionType(action_str)
        override_po_line_no = data.get("override_po_line_no")
        notes = data.get("notes", "")
        manual_edits = data.get("manual_edits")
        reviewer_name = data.get("reviewer_name", "AP Reviewer")

        updated = StagingService.apply_staged_review_action(
            staging_id=staging_id,
            line_no=line_no,
            action=action,
            override_po_line_no=override_po_line_no,
            notes=notes,
            manual_edits=manual_edits,
            reviewer_name=reviewer_name
        )

        lines_json = [json.loads(l.model_dump_json()) for l in updated["lines"]]
        audit_json = [json.loads(a.model_dump_json()) for a in updated["audit_log"]]
        summary_json = json.loads(updated["summary"].model_dump_json())

        payload = {
            "staging_id": updated["staging_id"],
            "invoice_id": updated["invoice_id"],
            "po_id": updated["po_id"],
            "vendor_name": updated["vendor_name"],
            "file_name": updated["file_name"],
            "status": updated["status"],
            "summary": summary_json,
            "lines": lines_json,
            "audit_log": audit_json,
            "updated_at": updated["updated_at"]
        }
        return JSONResponse(payload)

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Review action error: {str(e)}"}, status_code=500)


async def commit_staged_draft(request: Request):
    """Permanently commits an approved staged draft into the committed_invoices ledger."""
    try:
        staging_id = request.path_params.get("staging_id")
        data = await request.json() if request.headers.get("content-length", "0") != "0" else {}
        reviewer_name = data.get("reviewer_name", "AP Lead")
        notes = data.get("notes", "")

        committed_record = StagingService.commit_staged_draft(
            staging_id=staging_id,
            reviewer_name=reviewer_name,
            notes=notes
        )
        return JSONResponse(committed_record)

    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Commit error: {str(e)}"}, status_code=500)


async def list_committed_ledger(request: Request):
    """Lists all permanently committed reconciliation transactions."""
    commits = StagingRepository.list_committed_invoices()
    return JSONResponse({"total_commits": len(commits), "commits": commits})


async def list_master_pos(request: Request):
    """Lists all master Purchase Orders in the database."""
    pos = PORepository.list_all_pos()
    return JSONResponse({"total_pos": len(pos), "purchase_orders": pos})


# ── 2. Fallback & Live Simulation Endpoints (100% Backward Compatible) ────────

async def list_demo_scenarios(request: Request):
    """Returns available pre-configured demo invoice scenarios covering all 9 test categories."""
    if not SIM_INVOICES_PATH.exists():
        return JSONResponse({"error": "Simulation dataset not found."}, status_code=404)

    with open(SIM_INVOICES_PATH, encoding="utf-8") as f:
        invoices = json.load(f)

    scenarios_summary = []
    for inv in invoices:
        scenarios_summary.append({
            "invoice_id": inv["invoice_id"],
            "po_id": inv["po_id"],
            "vendor_name": inv["vendor_name"],
            "invoice_date": inv["invoice_date"],
            "scenario_category": inv.get("scenario_category", "standard_clean"),
            "scenario_description": inv.get("scenario_description", ""),
            "line_count": len(inv["invoice_lines"]),
            "po_line_count": len(inv["po_lines"])
        })

    return JSONResponse({
        "total_scenarios": len(scenarios_summary),
        "scenarios": scenarios_summary
    })


async def get_demo_scenario(request: Request):
    """Returns full Invoice and PO payload for a specific demo scenario."""
    invoice_id = request.path_params.get("invoice_id")
    if not SIM_INVOICES_PATH.exists():
        return JSONResponse({"error": "Simulation dataset not found."}, status_code=404)

    with open(SIM_INVOICES_PATH, encoding="utf-8") as f:
        invoices = json.load(f)

    target = next((inv for inv in invoices if inv["invoice_id"] == invoice_id), None)
    if not target:
        return JSONResponse({"error": f"Invoice ID '{invoice_id}' not found."}, status_code=404)

    invoice_obj = {
        "invoice_id": target["invoice_id"],
        "po_id": target["po_id"],
        "vendor_name": target["vendor_name"],
        "invoice_date": target["invoice_date"],
        "scenario_category": target.get("scenario_category", "standard_clean"),
        "scenario_description": target.get("scenario_description", ""),
        "invoice_lines": target["invoice_lines"]
    }

    po_obj = {
        "po_id": target["po_id"],
        "vendor_name": target["vendor_name"],
        "po_date": target["invoice_date"],
        "po_lines": target["po_lines"]
    }

    return JSONResponse({
        "invoice": invoice_obj,
        "purchase_order": po_obj
    })


async def run_reconciliation(request: Request):
    """Executes hybrid matching, greedy 1:1 assignment, numeric verification, and confidence gating.
    Also persists the result to SQLite staging so the commit endpoint can find it by staging_id.
    """
    try:
        data = await request.json()
        inv_data = data.get("invoice", {})
        po_data = data.get("purchase_order", {})

        invoice = Invoice(**inv_data)
        po = PurchaseOrder(**po_data)

        res = _engine.reconcile(invoice, po)

        # Persist to SQLite so commit endpoint can find it
        staging_id = StagingRepository.create_staged_reconciliation(
            rec_response=res,
            file_name=f"demo_{res.invoice_id}.json"
        )

        payload = json.loads(res.model_dump_json())
        payload["staging_id"] = staging_id
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": f"Reconciliation error: {str(e)}"}, status_code=500)


async def submit_human_review(request: Request):
    """Applies human reviewer action to in-memory reconciliation engine."""
    try:
        data = await request.json()
        req = ReviewActionRequest(**data)
        updated_res = _engine.apply_review_action(req)
        return JSONResponse(json.loads(updated_res.model_dump_json()))
    except ValueError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Review error: {str(e)}"}, status_code=500)


async def get_session_audit_log(request: Request):
    """Returns audit log of all automated actions and human decisions."""
    logs = [json.loads(log.model_dump_json()) for log in _engine.audit_log]
    return JSONResponse(logs)


async def run_batch_simulation_benchmark(request: Request):
    """Executes reconciliation across all 24 simulation invoices and returns overall benchmark metrics."""
    if not SIM_INVOICES_PATH.exists():
        return JSONResponse({"error": "Simulation dataset not found."}, status_code=404)

    with open(SIM_INVOICES_PATH, encoding="utf-8") as f:
        invoices = json.load(f)

    total_lines = 0
    first_pass_clean = 0
    ambiguous_lines = 0
    unmatched_lines = 0
    discrepancies = 0
    scenarios_results = []

    for raw_inv in invoices:
        inv = Invoice(
            invoice_id=raw_inv["invoice_id"],
            po_id=raw_inv["po_id"],
            vendor_name=raw_inv["vendor_name"],
            invoice_date=raw_inv["invoice_date"],
            scenario_category=raw_inv.get("scenario_category", ""),
            scenario_description=raw_inv.get("scenario_description", ""),
            invoice_lines=raw_inv["invoice_lines"]
        )
        po = PurchaseOrder(
            po_id=raw_inv["po_id"],
            vendor_name=raw_inv["vendor_name"],
            po_lines=raw_inv["po_lines"]
        )

        res = _engine.reconcile(inv, po)
        total_lines += len(res.lines)
        first_pass_clean += res.summary.confidently_matched_count
        ambiguous_lines += res.summary.ambiguous_count
        unmatched_lines += res.summary.unmatched_count
        discrepancies += res.summary.discrepancies_count

        scenarios_results.append({
            "invoice_id": res.invoice_id,
            "category": res.scenario_category,
            "summary": json.loads(res.summary.model_dump_json())
        })

    first_pass_pct = round((first_pass_clean / total_lines) * 100, 1) if total_lines > 0 else 0.0
    manual_intervention_pct = round(((ambiguous_lines + unmatched_lines) / total_lines) * 100, 1) if total_lines > 0 else 0.0

    return JSONResponse({
        "total_invoices_tested": len(invoices),
        "total_invoice_lines": total_lines,
        "first_pass_match_rate_pct": first_pass_pct,
        "confidently_matched_count": first_pass_clean,
        "lines_requiring_human_review_count": ambiguous_lines + unmatched_lines,
        "lines_requiring_human_review_pct": manual_intervention_pct,
        "discrepancies_flagged_count": discrepancies,
        "scenarios": scenarios_results
    })
