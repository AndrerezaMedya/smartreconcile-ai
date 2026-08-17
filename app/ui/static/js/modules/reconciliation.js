/**
 * SmartReconcile AI — Reconciliation Workspace & Row Item Rendering
 */

import { state } from "./state.js";
import { fetchScenarios, fetchScenarioDetail, postReconcile } from "./api.js";
import { formatNumber, formatFlagLabel, escapeHtml, getCategoryIcon, formatCategoryLabel } from "./ui.js";

let onOpenReviewModalCallback = null;
let onRefreshAuditCountCallback = null;

export function setOpenReviewModalHandler(handler) {
  onOpenReviewModalCallback = handler;
}

export function setRefreshAuditCountHandler(handler) {
  onRefreshAuditCountCallback = handler;
}

export async function loadScenarios() {
  const scenarioMetaDesc = document.getElementById("scenarioMetaDesc");
  try {
    const data = await fetchScenarios();
    state.currentScenarios = data.scenarios;
    renderScenarioChips();
    if (state.currentScenarios.length > 0) {
      loadAndReconcileScenario(state.currentScenarios[0].invoice_id);
    }
  } catch (err) {
    console.error("Failed to load scenarios:", err);
    if (scenarioMetaDesc) scenarioMetaDesc.textContent = "Error loading scenarios.";
  }
}

export function getFilteredScenarios() {
  if (state.currentFilter === "all") return state.currentScenarios;
  if (state.currentFilter === "discrepancy") {
    return state.currentScenarios.filter(s => ["price_variance", "quantity_variance", "uom_discrepancy", "math_error"].includes(s.scenario_category));
  }
  if (state.currentFilter === "exception") {
    return state.currentScenarios.filter(s => ["unmatched_invoice_line", "partial_delivery"].includes(s.scenario_category));
  }
  return state.currentScenarios.filter(s => s.scenario_category === state.currentFilter);
}

export function renderScenarioChips() {
  const chipsContainer = document.getElementById("scenarioChipsContainer");
  if (!chipsContainer) return;
  chipsContainer.innerHTML = "";
  const filtered = getFilteredScenarios();
  filtered.forEach((sc) => {
    const chip = document.createElement("div");
    chip.className = `scenario-chip ${sc.invoice_id === state.currentActiveInvoiceId ? "active" : ""}`;
    chip.innerHTML = `
      <span>${getCategoryIcon(sc.scenario_category)}</span>
      <span style="font-weight: 600;">${sc.invoice_id}</span>
      <span style="color: var(--text-dim);">•</span>
      <span style="color: var(--text-muted);">${formatCategoryLabel(sc.scenario_category)}</span>
    `;
    chip.addEventListener("click", () => loadAndReconcileScenario(sc.invoice_id));
    chipsContainer.appendChild(chip);
  });
}

export async function loadAndReconcileScenario(invoiceId) {
  state.currentActiveInvoiceId = invoiceId;
  renderScenarioChips();

  const scenarioMetaDesc = document.getElementById("scenarioMetaDesc");
  const sc = state.currentScenarios.find(s => s.invoice_id === invoiceId);
  if (sc && scenarioMetaDesc) {
    scenarioMetaDesc.innerHTML = `<strong style="color: #fff;">${escapeHtml(sc.vendor_name)}</strong> • ${escapeHtml(sc.scenario_description || formatCategoryLabel(sc.scenario_category))}`;
  }

  const tableBody = document.getElementById("reconciliationTableBody");
  if (tableBody) {
    tableBody.innerHTML = `
      <div style="padding: 32px; text-align: center; color: var(--text-muted); display: flex; align-items: center; justify-content: center; gap: 8px;">
        <svg class="nav-svg-icon spinner-icon" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
        <span>Executing hybrid lexical + MiniLM v2 matching on CPU...</span>
      </div>
    `;
  }

  try {
    const detailData = await fetchScenarioDetail(invoiceId);
    const recData = await postReconcile(detailData);
    state.currentStagedData = recData;
    renderReconciliationView(recData);
  } catch (err) {
    if (tableBody) {
      tableBody.innerHTML = `<div style="padding: 24px; color: var(--rose);">Error running reconciliation: ${err.message}</div>`;
    }
  }
}

export function renderReconciliationView(data) {
  const docVendorName = document.getElementById("docVendorName");
  const docInvoiceId = document.getElementById("docInvoiceId");
  const docPoId = document.getElementById("docPoId");
  const stagingIdBadge = document.getElementById("stagingIdBadge");
  const docOverallStatus = document.getElementById("docOverallStatus");
  const workflowStateText = document.getElementById("workflowStateText");

  if (docVendorName) docVendorName.textContent = data.vendor_name || "Supplier Invoice";
  if (docInvoiceId) docInvoiceId.textContent = data.invoice_id || "--";
  if (docPoId) docPoId.textContent = data.po_id || "--";

  const stagingId = data.staging_id || `STG-${data.invoice_id.replace(/[^A-Z0-9]/gi, "")}`;
  if (stagingIdBadge) stagingIdBadge.textContent = stagingId;

  const isClean = data.summary.overall_status === "CLEAN_AUTO_ACCEPT";
  const isCommitted = data.status === "COMMITTED";

  if (isCommitted) {
    if (docOverallStatus) {
      docOverallStatus.className = "badge badge-matched";
      docOverallStatus.textContent = "COMMITTED TO DATABASE";
    }
    if (workflowStateText) {
      workflowStateText.style.color = "var(--emerald)";
      workflowStateText.textContent = "Committed to Local SQLite Ledger";
    }
  } else if (isClean) {
    if (docOverallStatus) {
      docOverallStatus.className = "badge badge-matched";
      docOverallStatus.textContent = "AI: HIGH CONFIDENCE MATCH";
    }
    if (workflowStateText) {
      workflowStateText.style.color = "var(--emerald)";
      workflowStateText.textContent = "Pending Review — Ready for Reviewer Confirmation & Commit";
    }
  } else {
    if (docOverallStatus) {
      docOverallStatus.className = "badge badge-ambiguous";
      docOverallStatus.textContent = "AI: EXCEPTIONS DETECTED";
    }
    if (workflowStateText) {
      workflowStateText.style.color = "var(--amber)";
      workflowStateText.textContent = "Pending Review — Reviewer decision required before commit";
    }
  }

  // Update Top KPIs
  const kpiFirstPassRate = document.getElementById("kpiFirstPassRate");
  const kpiFirstPassSub = document.getElementById("kpiFirstPassSub");
  const kpiPrematched = document.getElementById("kpiPrematched");
  const kpiExceptions = document.getElementById("kpiExceptions");
  const sidebarExceptionBadge = document.getElementById("sidebarExceptionBadge");

  if (kpiFirstPassRate) kpiFirstPassRate.textContent = `${data.summary.first_pass_rate_pct.toFixed(1)}%`;
  if (kpiFirstPassSub) kpiFirstPassSub.textContent = isClean ? "All lines auto-verified clean" : "Automated first-pass alignment";
  if (kpiPrematched) kpiPrematched.textContent = `${data.summary.confidently_matched_count} / ${data.summary.total_invoice_lines}`;

  const pendingExceptions = (data.summary.ambiguous_count + data.summary.unmatched_count);
  if (kpiExceptions) kpiExceptions.textContent = `${pendingExceptions} lines`;
  if (sidebarExceptionBadge) sidebarExceptionBadge.textContent = pendingExceptions;

  const tableBody = document.getElementById("reconciliationTableBody");
  if (tableBody) {
    tableBody.innerHTML = "";
    data.lines.forEach((line) => {
      const row = document.createElement("div");
      row.className = "recon-row";
      row.innerHTML = buildRowHTML(line);
      tableBody.appendChild(row);
    });

    // Attach modal triggers
    tableBody.querySelectorAll(".btn-resolve-exception").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const lineNo = parseInt(e.currentTarget.getAttribute("data-line-no"));
        if (onOpenReviewModalCallback) onOpenReviewModalCallback(lineNo);
      });
    });
  }

  if (onRefreshAuditCountCallback) onRefreshAuditCountCallback();
}

export function buildRowHTML(line) {
  const v = line.verification || {};
  const hasDiscrepancy = v.has_discrepancy;
  const isReviewed = line.is_reviewed;

  // 1. Left Column: Billed Invoice Line
  const leftHtml = `
    <div class="col-invoice">
      <div class="line-num">#${line.line_no}</div>
      <div class="inv-details">
        <div class="item-desc-text">${escapeHtml(line.invoice_description)}</div>
        <div class="item-commercial-row">
          <span class="font-mono" style="color: #fff; font-weight: 600;">${line.invoice_qty} ${escapeHtml(line.invoice_uom)}</span>
          <span>@</span>
          <span class="font-mono">Rp ${formatNumber(line.invoice_unit_price)}</span>
          <span style="color: var(--text-dim);">•</span>
          <span class="font-mono" style="color: var(--text-secondary);">Total: <strong style="color: #fff;">Rp ${formatNumber(line.invoice_line_total)}</strong></span>
        </div>
      </div>
    </div>
  `;

  // 2. Center Column: AI Confidence & Method
  let statusBadgeClass = "badge-matched";
  let statusText = "MATCHED";

  if (line.status === "AMBIGUOUS") {
    statusBadgeClass = "badge-ambiguous";
    statusText = "AMBIGUOUS";
  } else if (line.status === "UNMATCHED") {
    statusBadgeClass = "badge-unmatched";
    statusText = "UNMATCHED";
  }

  const routeTag = line.is_semantic_routed
    ? `<span class="route-tag semantic" title="Triggered MiniLM-L12-v2 semantic reranking">SEMANTIC ROUTED</span>`
    : `<span class="route-tag lexical" title="Resolved via deterministic lexical attribute scoring">LEXICAL DIRECT</span>`;

  const centerHtml = `
    <div class="col-ai">
      <div class="ai-status-row">
        <span class="badge ${statusBadgeClass}">${statusText}</span>
        ${routeTag}
      </div>
      <div class="ai-meta-row">
        <span>Score: <strong>${(line.score || 0).toFixed(2)}</strong></span>
        <span>Margin: <strong>+${(line.confidence_margin || 0).toFixed(2)}</strong></span>
      </div>
    </div>
  `;

  // 3. Right Column: Matched PO Line & 4-Way Verification
  let rightHtml = "";
  if (line.assigned_po_line_no) {
    let flagsHtml = "";
    if (v.qty_flag && v.qty_flag !== "QTY_MATCH") {
      flagsHtml += `<span class="flag-chip flag-var">QTY: ${formatFlagLabel(v.qty_flag)}</span>`;
    }
    if (v.price_flag && v.price_flag !== "PRICE_MATCH") {
      flagsHtml += `<span class="flag-chip flag-var">PRICE: ${formatFlagLabel(v.price_flag)}</span>`;
    }
    if (v.uom_flag && v.uom_flag !== "UOM_MATCH") {
      flagsHtml += `<span class="flag-chip flag-var">UOM: ${formatFlagLabel(v.uom_flag)}</span>`;
    }
    if (v.math_flag && v.math_flag !== "MATH_CORRECT") {
      flagsHtml += `<span class="flag-chip flag-var">MATH: ${formatFlagLabel(v.math_flag)}</span>`;
    }

    const reviewStateBadge = isReviewed
      ? `<span class="flag-chip flag-ok font-mono">✓ ${line.review_action || "REVIEWED"}</span>`
      : "";

    rightHtml = `
      <div class="col-po">
        <div class="po-details">
          <div class="po-title-row">
            <span class="po-tag">PO #${line.assigned_po_line_no}</span>
            <span class="item-desc-text" style="font-size: 12px;">${escapeHtml(line.assigned_po_description || "")}</span>
            ${reviewStateBadge}
          </div>
          <div class="item-commercial-row">
            <span class="font-mono">PO: ${line.assigned_po_qty} ${escapeHtml(line.assigned_po_uom || "")} @ Rp ${formatNumber(line.assigned_po_unit_price)}</span>
          </div>
          ${flagsHtml ? `<div class="po-flags-row">${flagsHtml}</div>` : `<div class="po-flags-row"><span class="flag-chip flag-ok font-mono">✓ 4-WAY VERIFIED CLEAN</span></div>`}
        </div>
        <div>
          ${hasDiscrepancy || line.status !== "MATCHED" ? `
            <button class="btn btn-warning btn-sm btn-resolve-exception" data-line-no="${line.line_no}">
              <svg class="nav-svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              <span>Resolve</span>
            </button>
          ` : `
            <span class="font-mono" style="color: var(--emerald); font-size: 11px; font-weight: 600;">MATCH OK</span>
          `}
        </div>
      </div>
    `;
  } else {
    rightHtml = `
      <div class="col-po">
        <div class="po-details">
          <span class="flag-chip flag-var font-mono">NO AUTHORIZED PO LINE FOUND</span>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Potential unauthorized fee or extra freight surcharge.</div>
        </div>
        <button class="btn btn-danger btn-sm btn-resolve-exception" data-line-no="${line.line_no}">
          <svg class="nav-svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>
          <span>Dispute Line</span>
        </button>
      </div>
    `;
  }

  return leftHtml + centerHtml + rightHtml;
}

export function renderReviewQueueView(data) {
  const reviewQueueTableBody = document.getElementById("reviewQueueTableBody");
  if (!reviewQueueTableBody) return;
  reviewQueueTableBody.innerHTML = "";
  const exceptionLines = data.lines.filter(l => l.status !== "MATCHED" || (l.verification && l.verification.has_discrepancy));

  if (exceptionLines.length === 0) {
    reviewQueueTableBody.innerHTML = `
      <div style="padding: 32px; text-align: center; color: var(--emerald); font-weight: 600;">
        ✓ Zero pending exceptions in this invoice. All lines are confidently matched and verified clean.
      </div>
    `;
    return;
  }

  exceptionLines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "recon-row";
    row.innerHTML = buildRowHTML(line);
    reviewQueueTableBody.appendChild(row);
  });

  reviewQueueTableBody.querySelectorAll(".btn-resolve-exception").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const lineNo = parseInt(e.currentTarget.getAttribute("data-line-no"));
      if (onOpenReviewModalCallback) onOpenReviewModalCallback(lineNo);
    });
  });
}
