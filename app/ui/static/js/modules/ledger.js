/**
 * SmartReconcile AI — Commit Authorization & Confirmation
 */

import { state } from "./state.js";
import { postCommit } from "./api.js";
import { formatNumber, escapeHtml } from "./ui.js";
import { renderReconciliationView } from "./reconciliation.js";

export function openCommitModal() {
  if (!state.currentStagedData) return;

  // ── FR-25: Block commit if any UNMATCHED line has not been reviewed ──────────
  const unmatchedUnresolved = state.currentStagedData.lines.filter(
    l => l.status === "UNMATCHED" && !l.is_reviewed
  );
  if (unmatchedUnresolved.length > 0) {
    const lineNums = unmatchedUnresolved.map(l => `#${l.line_no}`).join(", ");
    const lineWord = unmatchedUnresolved.length === 1 ? "line" : "lines";
    alert(
      `⛔ Commit Blocked — Unresolved Exception${unmatchedUnresolved.length > 1 ? "s" : ""}\n\n` +
      `${unmatchedUnresolved.length} invoice ${lineWord} still have status UNMATCHED and have not been reviewed:\n` +
      `  Line(s): ${lineNums}\n\n` +
      `Please open the Review panel for each flagged line and either:\n` +
      `  • Approve or Override it to a valid PO line, or\n` +
      `  • Reject it (mark as exception)\n\n` +
      `All lines must be reviewed before committing to the ledger.`
    );
    return; // ← halt — do NOT open commit modal
  }
  // ────────────────────────────────────────────────────────────────────────────

  // Reset to the authorization form in case this modal previously showed a
  // success state for an earlier invoice in the same session.
  const formBody = document.getElementById("commitModalFormBody");
  const successBody = document.getElementById("commitModalSuccessBody");
  const formFooter = document.getElementById("commitModalFormFooter");
  const successFooter = document.getElementById("commitModalSuccessFooter");
  if (formBody) formBody.style.display = "block";
  if (successBody) successBody.style.display = "none";
  if (formFooter) formFooter.style.display = "flex";
  if (successFooter) successFooter.style.display = "none";

  const invId = state.currentStagedData.invoice_id || "--";
  const poId = state.currentStagedData.po_id || "--";
  const vendor = state.currentStagedData.vendor_name || "Authorized Supplier";

  const commitModalInvId = document.getElementById("commitModalInvId");
  const commitModalPoId = document.getElementById("commitModalPoId");
  if (commitModalInvId) commitModalInvId.textContent = invId;
  if (commitModalPoId) commitModalPoId.textContent = poId;

  const totalInv = state.currentStagedData.lines.reduce((sum, l) => sum + (l.invoice_line_total || 0), 0);
  const totalMatched = state.currentStagedData.lines.reduce((sum, l) => {
    return sum + (l.assigned_po_line_no ? ((l.assigned_po_qty || 0) * (l.assigned_po_unit_price || 0)) : 0);
  }, 0);

  const pendingExceptions = (state.currentStagedData.summary.ambiguous_count + state.currentStagedData.summary.unmatched_count);

  const commitSummaryBox = document.getElementById("commitSummaryBox");
  if (commitSummaryBox) {
    commitSummaryBox.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; font-size: 11.5px;">
        <div style="background: var(--bg-card-subtle); padding: 8px 10px; border-radius: var(--radius-xs); border: 1px solid var(--border-subtle);">
          <span style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 2px;">Invoice & PO Reference</span>
          <strong class="font-mono text-white">${escapeHtml(invId)}</strong> <span style="color: var(--text-dim);">→</span> <strong class="font-mono">${escapeHtml(poId)}</strong>
        </div>
        <div style="background: var(--bg-card-subtle); padding: 8px 10px; border-radius: var(--radius-xs); border: 1px solid var(--border-subtle);">
          <span style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 2px;">Supplier / Vendor</span>
          <strong style="color: #fff;">${escapeHtml(vendor)}</strong>
        </div>
      </div>
      <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--divider); font-size: 12px;">
        <span>Total Billed Invoiced Amount:</span> <strong class="font-mono text-white">Rp ${formatNumber(totalInv)}</strong>
      </div>
      <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--divider); font-size: 12px;">
        <span>Authorized Match Amount (PO Verified):</span> <strong class="font-mono text-emerald" style="font-size: 13px;">Rp ${formatNumber(totalMatched)}</strong>
      </div>
      <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--divider); font-size: 12px;">
        <span>Total Line Items:</span> <strong class="font-mono">${state.currentStagedData.lines.length} lines</strong>
      </div>
      <div style="display: flex; justify-content: space-between; padding-top: 6px; font-size: 12px;">
        <span>Pending Review Exceptions:</span> <strong class="font-mono" style="color: ${pendingExceptions > 0 ? 'var(--amber)' : 'var(--emerald)'};">${pendingExceptions > 0 ? pendingExceptions + ' pending' : '0 (All Resolved)'}</strong>
      </div>
    `;
  }

  const commitModal = document.getElementById("commitModal");
  if (commitModal) commitModal.style.display = "flex";
}

export async function submitCommitToDatabase() {
  if (!state.currentStagedData) return;
  const reviewerInput = document.getElementById("commitReviewerName");
  const notesInput = document.getElementById("commitNotesInput");
  const reviewerName = reviewerInput ? (reviewerInput.value.trim() || "Senior AP Specialist") : "Senior AP Specialist";
  const notes = notesInput ? (notesInput.value.trim() || "Commercial reconciliation approved for payment") : "Commercial reconciliation approved for payment";

  const stagingId = state.currentStagedData.staging_id || `STG-${state.currentStagedData.invoice_id.replace(/[^A-Z0-9]/gi, "")}`;

  try {
    const commitResult = await postCommit(stagingId, { reviewer_name: reviewerName, notes: notes });

    state.currentStagedData.status = "COMMITTED";
    renderReconciliationView(state.currentStagedData);

    // Swap the modal from the authorization form to an inline confirmation of
    // THIS invoice's own commit result — no separate ledger view to navigate to.
    document.getElementById("commitResultId").textContent = commitResult.commit_id;
    document.getElementById("commitResultInvoice").textContent = commitResult.invoice_id;
    document.getElementById("commitResultAmount").textContent = `Rp ${formatNumber(commitResult.total_matched_amount)}`;
    document.getElementById("commitResultReviewer").textContent = reviewerName;
    document.getElementById("commitResultTimestamp").textContent = commitResult.commit_timestamp;

    document.getElementById("commitModalFormBody").style.display = "none";
    document.getElementById("commitModalSuccessBody").style.display = "block";
    document.getElementById("commitModalFormFooter").style.display = "none";
    document.getElementById("commitModalSuccessFooter").style.display = "flex";
  } catch (err) {
    alert("Error committing to database: " + err.message);
  }
}
