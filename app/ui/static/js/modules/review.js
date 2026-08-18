/**
 * SmartReconcile AI — Human Review & Exception Decision Interface
 */

import { state } from "./state.js";
import { postReviewAction, postStagingReviewAction } from "./api.js";
import { formatNumber, formatFlagLabel, escapeHtml } from "./ui.js";
import { renderReconciliationView, renderReviewQueueView } from "./reconciliation.js";

export function openReviewModal(lineNo) {
  if (!state.currentStagedData) return;
  const line = state.currentStagedData.lines.find(l => l.line_no === lineNo);
  if (!line) return;

  state.activeReviewLine = line;
  state.selectedOverridePoNo = line.assigned_po_line_no;

  const invId = state.currentStagedData.invoice_id || "--";
  const poId = state.currentStagedData.po_id || "--";
  const v = line.verification || {};
  const hasDiscrepancy = v.has_discrepancy || line.status !== "MATCHED";

  // Header Elements
  const modalTitle = document.getElementById("modalLineTitle");
  if (modalTitle) modalTitle.textContent = `RECONCILIATION REVIEW • LINE #${line.line_no}`;

  const headerStatus = document.getElementById("modalHeaderStatus");
  if (headerStatus) {
    headerStatus.className = hasDiscrepancy ? "badge badge-ambiguous font-mono" : "badge badge-matched font-mono";
    headerStatus.textContent = hasDiscrepancy ? "AI: EXCEPTIONS DETECTED" : "AI: HIGH CONFIDENCE MATCH";
  }

  const headerSub = document.getElementById("modalHeaderSub");
  if (headerSub) {
    headerSub.textContent = `${invId} → ${poId} • Reviewer decision required before commit`;
  }

  // 1. SECTION 1: Side-by-Side Comparison
  const compGrid = document.getElementById("modalComparisonGrid");
  if (compGrid) {
    const invQty = line.invoice_qty || 0;
    const poQty = line.assigned_po_qty || 0;
    const invPrice = line.invoice_unit_price || 0;
    const poPrice = line.assigned_po_unit_price || 0;
    const invTotal = line.invoice_line_total || (invQty * invPrice);
    const poTotal = (poQty * poPrice);

    // Variances
    let qtyDeltaHtml = `<span class="delta-badge ok">✓ MATCH</span>`;
    if (poQty > 0 && Math.abs(invQty - poQty) > 0.001) {
      const qtyVarPct = ((invQty - poQty) / poQty) * 100;
      qtyDeltaHtml = `<span class="delta-badge ${qtyVarPct > 0 ? 'err' : 'warn'}">${qtyVarPct > 0 ? '+' : ''}${qtyVarPct.toFixed(1)}% (${(invQty - poQty) > 0 ? '+' : ''}${(invQty - poQty).toFixed(1)})</span>`;
    }

    let priceDeltaHtml = `<span class="delta-badge ok">✓ MATCH</span>`;
    if (poPrice > 0 && Math.abs(invPrice - poPrice) > (poPrice * 0.01)) {
      const priceVarPct = ((invPrice - poPrice) / poPrice) * 100;
      priceDeltaHtml = `<span class="delta-badge ${priceVarPct > 0 ? 'err' : 'warn'}">${priceVarPct > 0 ? '+' : ''}${priceVarPct.toFixed(1)}% (Rp ${formatNumber(invPrice - poPrice)})</span>`;
    }

    let totalDeltaHtml = `<span class="delta-badge ok">✓ EQUAL</span>`;
    if (Math.abs(invTotal - poTotal) > 1.0) {
      totalDeltaHtml = `<span class="delta-badge err">Δ Rp ${formatNumber(invTotal - poTotal)}</span>`;
    }

    let uomDeltaHtml = `<span class="delta-badge ok">✓ MATCH</span>`;
    if (line.assigned_po_uom && line.invoice_uom.toLowerCase() !== line.assigned_po_uom.toLowerCase()) {
      uomDeltaHtml = `<span class="delta-badge err">MISMATCH (${escapeHtml(line.invoice_uom)} vs ${escapeHtml(line.assigned_po_uom)})</span>`;
    }

    compGrid.innerHTML = `
      <!-- Left: Billed Invoice -->
      <div class="comp-side-box">
        <div class="comp-side-header">
          <span>Billed Invoice Item</span>
          <span class="font-mono">#${line.line_no}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Description:</span>
          <span class="comp-val font-mono" style="font-weight: 700; color: #fff;">${escapeHtml(line.invoice_description)}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Billed Quantity:</span>
          <span class="comp-val font-mono">${line.invoice_qty} ${escapeHtml(line.invoice_uom)}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Billed Unit Price:</span>
          <span class="comp-val font-mono">Rp ${formatNumber(line.invoice_unit_price)}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Billed Line Total:</span>
          <span class="comp-val font-mono" style="color: #fff; font-weight: 700;">Rp ${formatNumber(invTotal)}</span>
        </div>
      </div>

      <!-- Right: Matched PO Line -->
      <div class="comp-side-box">
        <div class="comp-side-header">
          <span>Matched PO Reference</span>
          <span class="font-mono">${line.assigned_po_line_no ? `PO Line #${line.assigned_po_line_no}` : 'NO PO LINE'}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Description:</span>
          <span class="comp-val font-mono" style="font-weight: 700; color: #fff;">${escapeHtml(line.assigned_po_description || 'Unmatched')}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">Ordered Quantity:</span>
          <span class="comp-val font-mono">${line.assigned_po_qty ? `${line.assigned_po_qty} ${escapeHtml(line.assigned_po_uom || '')}` : '-'} &nbsp; ${qtyDeltaHtml}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">PO Unit Price:</span>
          <span class="comp-val font-mono">${line.assigned_po_unit_price ? `Rp ${formatNumber(line.assigned_po_unit_price)}` : '-'} &nbsp; ${priceDeltaHtml}</span>
        </div>
        <div class="comp-row">
          <span class="comp-label">PO Line Total:</span>
          <span class="comp-val font-mono" style="color: var(--emerald); font-weight: 700;">Rp ${formatNumber(poTotal)} &nbsp; ${totalDeltaHtml}</span>
        </div>
      </div>
    `;
  }

  // 2. SECTION 2: AI Recommendation Panel
  const aiPanel = document.getElementById("modalAiRecPanel");
  if (aiPanel) {
    const routeTag = line.is_semantic_routed
      ? `<span class="route-tag semantic">SEMANTIC ROUTED (MiniLM CPU)</span>`
      : `<span class="route-tag lexical">LEXICAL DIRECT</span>`;

    aiPanel.innerHTML = `
      <div class="ai-rec-box">
        <div class="ai-rec-header">
          <span class="badge ${line.status === 'MATCHED' ? 'badge-matched' : (line.status === 'AMBIGUOUS' ? 'badge-ambiguous' : 'badge-unmatched')}">${line.status}</span>
          ${routeTag}
        </div>
        <div class="ai-rec-metrics">
          <span>Similarity: <strong class="text-white">${(line.score || 0).toFixed(2)}</strong></span>
          <span>Margin: <strong class="text-white">+${(line.confidence_margin || 0).toFixed(2)}</strong></span>
        </div>
        <div class="ai-rec-rationale">
          ${line.is_semantic_routed
            ? 'Recommended because this candidate achieved the highest similarity via MiniLM-L12-v2 multilingual semantic reranking.'
            : 'Recommended via direct deterministic lexical and specification attribute matching.'}
        </div>
      </div>
    `;
  }

  // 3. SECTION 3: Deterministic 4-Way Verification Strip
  const stripPanel = document.getElementById("modal4WayStrip");
  if (stripPanel) {
    const qtyClass = (!v.qty_flag || v.qty_flag === 'QTY_MATCH') ? 'ok' : 'err';
    const priceClass = (!v.price_flag || v.price_flag === 'PRICE_MATCH') ? 'ok' : 'err';
    const uomClass = (!v.uom_flag || v.uom_flag === 'UOM_MATCH') ? 'ok' : 'err';
    const mathClass = (!v.math_flag || v.math_flag === 'MATH_CORRECT') ? 'ok' : 'err';

    stripPanel.innerHTML = `
      <div class="strip-4way-grid">
        <div class="strip-pill ${qtyClass}">
          <span>QTY</span>
          <span>${v.qty_flag ? formatFlagLabel(v.qty_flag) : 'MATCH'}</span>
        </div>
        <div class="strip-pill ${priceClass}">
          <span>PRICE</span>
          <span>${v.price_flag ? formatFlagLabel(v.price_flag) : 'MATCH'}</span>
        </div>
        <div class="strip-pill ${uomClass}">
          <span>UOM</span>
          <span>${v.uom_flag ? formatFlagLabel(v.uom_flag) : 'MATCH'}</span>
        </div>
        <div class="strip-pill ${mathClass}">
          <span>MATH</span>
          <span>${v.math_flag ? formatFlagLabel(v.math_flag) : 'CORRECT'}</span>
        </div>
      </div>
    `;
  }

  // 4. SECTION 4: Alternative PO Candidates
  const candList = document.getElementById("modalCandidateList");
  if (candList) {
    candList.innerHTML = "";

    if (line.top_candidates && line.top_candidates.length > 0) {
      line.top_candidates.forEach((cand, idx) => {
        const isSelected = cand.po_line_no === state.selectedOverridePoNo;
        const card = document.createElement("div");
        card.className = `candidate-item-card ${isSelected ? "selected" : ""}`;
        card.innerHTML = `
          <div style="display: flex; align-items: center; gap: 10px;">
            <span class="candidate-rank-badge">#${idx + 1}</span>
            <div>
              <div style="font-size: 12px; font-weight: 600; color: #fff;">
                <span class="font-mono" style="color: var(--primary); margin-right: 4px;">[PO Line #${cand.po_line_no}]</span>
                ${escapeHtml(cand.description)}
              </div>
              <div class="font-mono" style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                Ordered: ${cand.ordered_qty} ${escapeHtml(cand.uom)} @ Rp ${formatNumber(cand.unit_price)}
              </div>
            </div>
          </div>
          <div style="text-align: right;">
            <div class="font-mono" style="font-weight: 700; color: var(--sky); font-size: 12px;">${cand.hybrid_score.toFixed(2)}</div>
            <div style="font-size: 9px; color: var(--text-dim); text-transform: uppercase;">Similarity</div>
          </div>
        `;

        card.addEventListener("click", () => {
          state.selectedOverridePoNo = cand.po_line_no;
          document.querySelectorAll(".candidate-item-card").forEach(c => c.classList.remove("selected"));
          card.classList.add("selected");
          updateConfirmButtonText();
        });

        candList.appendChild(card);
      });
    } else {
      candList.innerHTML = `<div style="font-size: 11.5px; color: var(--text-dim); padding: 8px;">No candidate PO lines available in master record.</div>`;
    }
  }

  // 5. Section 5: Decision button & notes
  updateConfirmButtonText();
  const notesInput = document.getElementById("modalReviewNotes");
  if (notesInput) notesInput.value = line.review_notes || "";

  const reviewModal = document.getElementById("reviewModal");
  if (reviewModal) reviewModal.style.display = "flex";
}

export function updateConfirmButtonText() {
  const confirmText = document.getElementById("btnModalConfirmText");
  if (!confirmText || !state.activeReviewLine) return;

  if (state.selectedOverridePoNo && state.selectedOverridePoNo !== state.activeReviewLine.assigned_po_line_no) {
    confirmText.textContent = `Override to PO Line #${state.selectedOverridePoNo}`;
  } else {
    confirmText.textContent = "Confirm & Approve";
  }
}

export async function submitReviewApprovalOrOverride() {
  if (!state.activeReviewLine || !state.currentStagedData) return;
  const lineNo = state.activeReviewLine.line_no;
  const notesInput = document.getElementById("modalReviewNotes");
  const notes = notesInput ? notesInput.value.trim() : "";

  let action = "APPROVE";
  let overridePoNo = null;

  if (state.selectedOverridePoNo && state.selectedOverridePoNo !== state.activeReviewLine.assigned_po_line_no) {
    action = "OVERRIDE";
    overridePoNo = state.selectedOverridePoNo;
  }

  const payload = {
    line_no: lineNo,
    action: action,
    override_po_line_no: overridePoNo,
    notes: notes || (action === "OVERRIDE" ? `Overridden to PO Line #${overridePoNo}` : "Approved by reviewer")
  };

  try {
    const stagingId = state.currentStagedData?.staging_id;
    // Prefer DB-backed staging review (persists to SQLite, captures audit trail).
    // Fall back to in-memory /api/review only if no staging_id is present.
    const updatedData = stagingId
      ? await postStagingReviewAction(stagingId, payload)
      : await postReviewAction(payload);

    // Restore staging_id only needed for in-memory fallback path
    if (!stagingId && state.currentStagedData?.staging_id) {
      updatedData.staging_id = state.currentStagedData.staging_id;
    }
    state.currentStagedData = updatedData;
    renderReconciliationView(updatedData);
    if (state.currentView === "viewReviewQueue") renderReviewQueueView(updatedData);
    const reviewModal = document.getElementById("reviewModal");
    if (reviewModal) reviewModal.style.display = "none";
  } catch (err) {
    alert("Error submitting review: " + err.message);
  }
}

export async function submitReviewRejection() {
  if (!state.activeReviewLine || !state.currentStagedData) return;
  const lineNo = state.activeReviewLine.line_no;
  const notesInput = document.getElementById("modalReviewNotes");
  const notes = notesInput ? notesInput.value.trim() : "";

  const payload = {
    line_no: lineNo,
    action: "REJECT",
    notes: notes || "Rejected unapproved charge / disputed line"
  };

  try {
    const stagingId = state.currentStagedData?.staging_id;
    const updatedData = stagingId
      ? await postStagingReviewAction(stagingId, payload)
      : await postReviewAction(payload);

    if (!stagingId && state.currentStagedData?.staging_id) {
      updatedData.staging_id = state.currentStagedData.staging_id;
    }
    state.currentStagedData = updatedData;
    renderReconciliationView(updatedData);
    if (state.currentView === "viewReviewQueue") renderReviewQueueView(updatedData);
    const reviewModal = document.getElementById("reviewModal");
    if (reviewModal) reviewModal.style.display = "none";
  } catch (err) {
    alert("Error rejecting line: " + err.message);
  }
}
