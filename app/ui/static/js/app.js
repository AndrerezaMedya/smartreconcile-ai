/**
 * SmartReconcile AI — Enterprise Product Frontend Application Bootstrap
 * Methodology: UI/UX Pro Max (High Density, Pure Modern Enterprise SaaS)
 */

import { state } from "./modules/state.js";
import { initSidebarNavigation, registerViewCallback } from "./modules/views.js";
import {
  loadScenarios,
  renderScenarioChips,
  getFilteredScenarios,
  loadAndReconcileScenario,
  renderReviewQueueView,
  setOpenReviewModalHandler
} from "./modules/reconciliation.js";
import {
  openReviewModal,
  submitReviewApprovalOrOverride,
  submitReviewRejection
} from "./modules/review.js";
import { initUploadDropzone } from "./modules/upload.js";
import {
  openCommitModal,
  submitCommitToDatabase
} from "./modules/ledger.js";

// ── 1. Lifecycle Bootstrap ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  // Wire inter-module handlers
  setOpenReviewModalHandler(openReviewModal);

  // Register view-switching hooks
  registerViewCallback("viewReviewQueue", () => {
    if (state.currentStagedData) renderReviewQueueView(state.currentStagedData);
  });

  // Initialize UI components
  initSidebarNavigation();
  initEventListeners();
  initUploadDropzone();
  loadScenarios();
});

// ── 2. Event Listeners ───────────────────────────────────────────────────────

function initEventListeners() {
  const btnReconcileCurrent = document.getElementById("btnReconcileCurrent");
  if (btnReconcileCurrent) {
    btnReconcileCurrent.addEventListener("click", () => {
      if (state.currentActiveInvoiceId) loadAndReconcileScenario(state.currentActiveInvoiceId);
    });
  }

  // Category Filter Pills
  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      state.currentFilter = pill.getAttribute("data-filter");
      renderScenarioChips();

      const filtered = getFilteredScenarios();
      if (filtered.length > 0) {
        loadAndReconcileScenario(filtered[0].invoice_id);
      }
    });
  });

  // Modal Close Controls
  const reviewModal = document.getElementById("reviewModal");
  const commitModal = document.getElementById("commitModal");

  const btnCloseModal = document.getElementById("btnCloseModal");
  const btnModalCancel = document.getElementById("btnModalCancel");
  if (btnCloseModal) btnCloseModal.addEventListener("click", () => { if (reviewModal) reviewModal.style.display = "none"; });
  if (btnModalCancel) btnModalCancel.addEventListener("click", () => { if (reviewModal) reviewModal.style.display = "none"; });

  const btnCloseCommitModal = document.getElementById("btnCloseCommitModal");
  const btnCancelCommit = document.getElementById("btnCancelCommit");
  const btnCloseCommitSuccess = document.getElementById("btnCloseCommitSuccess");
  if (btnCloseCommitModal) btnCloseCommitModal.addEventListener("click", () => { if (commitModal) commitModal.style.display = "none"; });
  if (btnCancelCommit) btnCancelCommit.addEventListener("click", () => { if (commitModal) commitModal.style.display = "none"; });
  if (btnCloseCommitSuccess) btnCloseCommitSuccess.addEventListener("click", () => { if (commitModal) commitModal.style.display = "none"; });

  // Commit Modal Actions
  const btnOpenCommitModal = document.getElementById("btnOpenCommitModal");
  if (btnOpenCommitModal) btnOpenCommitModal.addEventListener("click", openCommitModal);

  const btnConfirmCommit = document.getElementById("btnConfirmCommit");
  if (btnConfirmCommit) btnConfirmCommit.addEventListener("click", submitCommitToDatabase);

  // Review Modal Actions
  const btnModalConfirm = document.getElementById("btnModalConfirm");
  if (btnModalConfirm) btnModalConfirm.addEventListener("click", submitReviewApprovalOrOverride);

  const btnModalReject = document.getElementById("btnModalReject");
  if (btnModalReject) btnModalReject.addEventListener("click", submitReviewRejection);
}
