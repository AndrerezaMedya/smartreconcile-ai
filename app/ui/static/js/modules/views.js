/**
 * SmartReconcile AI — Multi-View Routing & Navigation
 */

import { state } from "./state.js";

const viewTitles = {
  viewReconcile: "Reconciliation Workspace",
  viewReviewQueue: "Review Queue & Exceptions",
  viewUpload: "Invoice File Ingestion",
  viewLedger: "Committed ERP Ledger",
  viewAudit: "Compliance Audit Trail"
};

let onViewSwitchCallbacks = {};

export function registerViewCallback(viewId, callback) {
  onViewSwitchCallbacks[viewId] = callback;
}

export function switchView(targetViewId) {
  document.querySelectorAll(".sidebar-item").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".view-container").forEach(v => v.style.display = "none");

  const sidebarBtn = document.querySelector(`.sidebar-item[data-view="${targetViewId}"]`);
  if (sidebarBtn) sidebarBtn.classList.add("active");

  state.currentView = targetViewId;

  const targetView = document.getElementById(targetViewId);
  if (targetView) targetView.style.display = "block";

  const breadcrumb = document.getElementById("currentBreadcrumb");
  if (breadcrumb) breadcrumb.textContent = viewTitles[targetViewId] || "Workspace";

  if (onViewSwitchCallbacks[targetViewId]) {
    onViewSwitchCallbacks[targetViewId]();
  }
}

export function initSidebarNavigation() {
  document.querySelectorAll(".sidebar-item").forEach(item => {
    item.addEventListener("click", () => {
      const targetViewId = item.getAttribute("data-view");
      switchView(targetViewId);
    });
  });
}
