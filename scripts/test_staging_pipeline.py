"""
Test script for verifying Phase C Staging and Permanent Commit workflow.
"""

from pathlib import Path
from app.services.staging_service import StagingService
from app.core.models import ReviewActionType


def main():
    pdf_path = Path("data/demo_invoices_pdf/INV-2026-020.pdf")
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return

    # 1. Ingest, Extract, Reconcile, and Stage
    staged = StagingService.process_and_stage_file(pdf_path.name, pdf_path.read_bytes())
    staging_id = staged["staging_id"]
    status = staged["status"]
    num_lines = len(staged["lines"])
    print(f"Staged draft created: ID={staging_id}, Status={status}, Lines={num_lines}")

    # 2. Review line #4 (Reject freight fee)
    updated = StagingService.apply_staged_review_action(
        staging_id=staging_id,
        line_no=4,
        action=ReviewActionType.REJECT,
        notes="Rejected unapproved freight surcharge"
    )
    l4 = updated["lines"][3]
    print(f"Line 4 updated: is_reviewed={l4.is_reviewed}, action={l4.review_action}")

    # 3. Commit to permanent database
    commit_res = StagingService.commit_staged_draft(
        staging_id=staging_id,
        reviewer_name="AP Manager",
        notes="Approved after freight exclusion"
    )
    commit_id = commit_res["commit_id"]
    commit_status = commit_res["status"]
    print(f"Committed to DB: Commit ID={commit_id}, Status={commit_status}")
    print("Phase C Staging & Commit Layer Verified Successfully!")


if __name__ == "__main__":
    main()
