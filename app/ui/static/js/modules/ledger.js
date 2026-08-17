/**
 * SmartReconcile AI — Committed Ledger, Audit Log & Simulation Benchmark
 * Methodology: UI/UX Pro Max (High Density, Enterprise Transaction History & Auditability)
 */

import { state } from "./state.js";
import { postCommit, fetchCommittedLedger, fetchAuditLog, postBenchmarkRunAll } from "./api.js";
import { formatNumber, escapeHtml, formatCategoryLabel } from "./ui.js";
import { renderReconciliationView } from "./reconciliation.js";

// Cached records for client-side search/filter
let cachedCommits = [];
let cachedAuditLogs = [];
let ledgerSearchBound = false;
let auditSearchBound = false;

// ── 1. Commit Confirmation Modal ─────────────────────────────────────────────

export function openCommitModal() {
  if (!state.currentStagedData) return;

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
    const commitModal = document.getElementById("commitModal");
    if (commitModal) commitModal.style.display = "none";

    state.currentStagedData.status = "COMMITTED";
    renderReconciliationView(state.currentStagedData);
    refreshLedgerCount();
    refreshAuditCount();

    alert(`✅ Transaction permanently committed to SQLite reconciliation ledger!\n\nCommit ID: ${commitResult.commit_id}\nAuthorized Amount: Rp ${formatNumber(commitResult.total_matched_amount)}\nReviewer: ${reviewerName}`);
  } catch (err) {
    alert("Error committing to database: " + err.message);
  }
}

// ── 2. Committed Transaction Ledger ──────────────────────────────────────────

export async function loadCommittedLedger() {
  const tbody = document.getElementById("committedLedgerTableBody");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 24px; color: var(--text-muted);">Loading committed transaction records from SQLite...</td></tr>`;

  // Bind real-time search if not yet bound
  if (!ledgerSearchBound) {
    const searchInput = document.getElementById("ledgerSearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        renderLedgerRows(e.target.value);
      });
      ledgerSearchBound = true;
    }
  }

  try {
    const data = await fetchCommittedLedger();
    cachedCommits = data.commits || [];

    const searchInput = document.getElementById("ledgerSearchInput");
    const query = searchInput ? searchInput.value : "";
    renderLedgerRows(query);

    const sidebarLedgerBadge = document.getElementById("sidebarLedgerBadge");
    if (sidebarLedgerBadge) sidebarLedgerBadge.textContent = cachedCommits.length;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" style="color: var(--rose); padding: 24px; text-align: center;">Error loading ledger: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function renderLedgerRows(query = "") {
  const tbody = document.getElementById("committedLedgerTableBody");
  const countBadge = document.getElementById("ledgerRecordCount");
  if (!tbody) return;

  const q = query.trim().toLowerCase();
  const filtered = q
    ? cachedCommits.filter(c => 
        (c.commit_id || "").toLowerCase().includes(q) ||
        (c.staging_id || "").toLowerCase().includes(q) ||
        (c.invoice_id || "").toLowerCase().includes(q) ||
        (c.po_id || "").toLowerCase().includes(q) ||
        (c.vendor_name || "").toLowerCase().includes(q) ||
        (c.reviewer_name || "").toLowerCase().includes(q)
      )
    : cachedCommits;

  if (countBadge) {
    countBadge.textContent = q ? `${filtered.length} of ${cachedCommits.length} records` : `${cachedCommits.length} committed`;
  }

  if (filtered.length === 0) {
    if (cachedCommits.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim); padding: 36px;">No committed invoices in the SQLite database yet. Reconcile an invoice and click "Commit Reconciliation".</td></tr>`;
    } else {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim); padding: 36px;">No records matching filter "<strong>${escapeHtml(q)}</strong>".</td></tr>`;
    }
    return;
  }

  tbody.innerHTML = filtered.map(c => `
    <tr>
      <td><strong class="font-mono text-white">${escapeHtml(c.commit_id)}</strong></td>
      <td><span class="badge badge-staging font-mono">${escapeHtml(c.staging_id)}</span></td>
      <td><strong class="font-mono" style="color: #93C5FD;">${escapeHtml(c.invoice_id)}</strong></td>
      <td><strong class="font-mono">${escapeHtml(c.po_id)}</strong></td>
      <td>${escapeHtml(c.vendor_name)}</td>
      <td class="font-mono" style="text-align: right;">Rp ${formatNumber(c.total_invoiced_amount)}</td>
      <td class="font-mono text-emerald" style="text-align: right; font-weight: 700;">Rp ${formatNumber(c.total_matched_amount)}</td>
      <td><span class="badge badge-reviewer">${escapeHtml(c.reviewer_name)}</span></td>
      <td class="font-mono" style="font-size: 11px; color: var(--text-dim);">${escapeHtml(c.commit_timestamp)}</td>
      <td><span class="badge badge-matched font-mono">COMMITTED</span></td>
    </tr>
  `).join("");
}

// ── 3. Compliance Audit Trail ────────────────────────────────────────────────

export async function loadAuditTrail() {
  const tbody = document.getElementById("auditTableBody");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 24px; color: var(--text-muted);">Loading compliance audit events...</td></tr>`;

  // Bind real-time search if not yet bound
  if (!auditSearchBound) {
    const searchInput = document.getElementById("auditSearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        renderAuditRows(e.target.value);
      });
      auditSearchBound = true;
    }
  }

  try {
    const logs = await fetchAuditLog();
    cachedAuditLogs = logs || [];

    const searchInput = document.getElementById("auditSearchInput");
    const query = searchInput ? searchInput.value : "";
    renderAuditRows(query);

    const sidebarAuditBadge = document.getElementById("sidebarAuditBadge");
    if (sidebarAuditBadge) sidebarAuditBadge.textContent = cachedAuditLogs.length;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="color: var(--rose); padding: 24px; text-align: center;">Error loading audit trail: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function getAuditActionBadge(action = "") {
  const act = action.toUpperCase();
  if (act.includes("AUTO") || act.includes("RECONCILE") || act.includes("MATCH")) {
    return `<span class="badge badge-sky font-mono">AI RECONCILE</span>`;
  } else if (act.includes("VERIF") || act.includes("RULE") || act.includes("CHECK")) {
    return `<span class="badge badge-indigo font-mono">RULE VERIFY</span>`;
  } else if (act.includes("APPROVE") || act.includes("CONFIRM")) {
    return `<span class="badge badge-matched font-mono">HUMAN APPROVE</span>`;
  } else if (act.includes("OVERRIDE") || act.includes("CHANGE")) {
    return `<span class="badge badge-purple font-mono">HUMAN OVERRIDE</span>`;
  } else if (act.includes("REJECT") || act.includes("DISPUTE")) {
    return `<span class="badge badge-rose font-mono">HUMAN DISPUTE</span>`;
  } else if (act.includes("COMMIT") || act.includes("LEDGER")) {
    return `<span class="badge badge-matched font-mono">COMMIT TRANSACTION</span>`;
  }
  return `<span class="badge badge-staging font-mono">${escapeHtml(action)}</span>`;
}

function renderAuditRows(query = "") {
  const tbody = document.getElementById("auditTableBody");
  const countBadge = document.getElementById("auditRecordCount");
  if (!tbody) return;

  const q = query.trim().toLowerCase();
  const filtered = q
    ? cachedAuditLogs.filter(l => 
        (l.action || "").toLowerCase().includes(q) ||
        (l.details || "").toLowerCase().includes(q) ||
        (l.user || "").toLowerCase().includes(q) ||
        (l.timestamp || "").toLowerCase().includes(q) ||
        String(l.line_no || "").includes(q)
      )
    : cachedAuditLogs;

  if (countBadge) {
    countBadge.textContent = q ? `${filtered.length} of ${cachedAuditLogs.length} events` : `${cachedAuditLogs.length} events`;
  }

  if (filtered.length === 0) {
    if (cachedAuditLogs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-dim); padding: 36px;">No audit events recorded yet. Perform a reconciliation or review action to generate compliance logs.</td></tr>`;
    } else {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-dim); padding: 36px;">No events matching filter "<strong>${escapeHtml(q)}</strong>".</td></tr>`;
    }
    return;
  }

  tbody.innerHTML = filtered.map(l => `
    <tr>
      <td class="font-mono" style="font-size: 11px; color: var(--text-dim);">${escapeHtml(l.timestamp)}</td>
      <td>${getAuditActionBadge(l.action)}</td>
      <td class="font-mono" style="font-weight: 600; color: #fff;">${l.line_no ? '#' + l.line_no : '—'}</td>
      <td style="color: var(--text-secondary); line-height: 1.35;">${escapeHtml(l.details)}</td>
      <td><strong style="color: #fff;">${escapeHtml(l.user)}</strong></td>
    </tr>
  `).join("");
}

// ── 4. Badges & Export ───────────────────────────────────────────────────────

export async function refreshLedgerCount() {
  try {
    const data = await fetchCommittedLedger();
    const sidebarLedgerBadge = document.getElementById("sidebarLedgerBadge");
    if (sidebarLedgerBadge) sidebarLedgerBadge.textContent = (data.commits || []).length;
  } catch (err) {}
}

export async function refreshAuditCount() {
  try {
    const logs = await fetchAuditLog();
    const sidebarAuditBadge = document.getElementById("sidebarAuditBadge");
    if (sidebarAuditBadge) sidebarAuditBadge.textContent = logs.length;
  } catch (err) {}
}

export function exportAuditLog() {
  fetchAuditLog()
    .then(data => {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `smartreconcile_audit_log_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
}

// ── 5. Batch Simulation Benchmark ───────────────────────────────────────────

export async function runBatchBenchmark() {
  const benchmarkModal = document.getElementById("benchmarkModal");
  if (benchmarkModal) benchmarkModal.style.display = "flex";

  const summaryGrid = document.getElementById("benchmarkSummaryGrid");
  const tbody = document.getElementById("benchmarkTableBody");

  if (summaryGrid) {
    summaryGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: var(--text-muted);">Running benchmark on all 24 invoices (106 line items) on CPU...</div>`;
  }
  if (tbody) tbody.innerHTML = "";

  try {
    const data = await postBenchmarkRunAll();

    if (summaryGrid) {
      summaryGrid.innerHTML = `
        <div class="kpi-card kpi-match">
          <div class="kpi-top"><span>First-Pass Match Rate</span><span>🎯</span></div>
          <div class="kpi-main-val font-mono">${data.first_pass_match_rate_pct.toFixed(1)}%</div>
          <div class="kpi-caption">Clean auto-matched</div>
        </div>
        <div class="kpi-card kpi-clean">
          <div class="kpi-top"><span>Pre-Matched Lines</span><span>✓</span></div>
          <div class="kpi-main-val font-mono">${data.confidently_matched_count} / ${data.total_invoice_lines}</div>
          <div class="kpi-caption">Zero commercial discrepancy</div>
        </div>
        <div class="kpi-card kpi-review">
          <div class="kpi-top"><span>Targeted Decisions</span><span>⚠️</span></div>
          <div class="kpi-main-val font-mono">${data.lines_requiring_human_review_count} lines</div>
          <div class="kpi-caption">Targeted exceptions (${data.lines_requiring_human_review_pct.toFixed(1)}%)</div>
        </div>
      `;
    }

    if (tbody) {
      tbody.innerHTML = data.scenarios.map(sc => `
        <tr>
          <td><strong class="font-mono" style="color: #fff;">${escapeHtml(sc.invoice_id)}</strong></td>
          <td>${formatCategoryLabel(sc.category)}</td>
          <td class="font-mono">${sc.summary.total_invoice_lines}</td>
          <td class="font-mono" style="color: var(--emerald);">${sc.summary.confidently_matched_count}</td>
          <td class="font-mono" style="color: var(--amber);">${sc.summary.ambiguous_count + sc.summary.unmatched_count}</td>
          <td class="font-mono">${sc.summary.discrepancies_count}</td>
          <td><span class="badge ${sc.summary.overall_status === 'CLEAN_AUTO_ACCEPT' ? 'badge-matched' : 'badge-ambiguous'}">${sc.summary.overall_status === 'CLEAN_AUTO_ACCEPT' ? 'CLEAN' : 'REVIEW'}</span></td>
        </tr>
      `).join("");
    }

  } catch (err) {
    if (summaryGrid) {
      summaryGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 24px; color: var(--rose);">Benchmark error: ${err.message}</div>`;
    }
  }
}
