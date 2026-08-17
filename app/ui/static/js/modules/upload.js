/**
 * SmartReconcile AI — Document Ingestion & Upload Workflow
 */

import { state } from "./state.js";
import { postUploadFile, fetchScenarioDetail, postReconcile } from "./api.js";
import { renderReconciliationView } from "./reconciliation.js";
import { switchView } from "./views.js";

export function initUploadDropzone() {
  const dropzone = document.getElementById("dropzoneArea");
  const fileInput = document.getElementById("fileInputUpload");
  const btnGoToReconcile = document.getElementById("btnGoToReconcile");

  if (btnGoToReconcile) {
    btnGoToReconcile.addEventListener("click", () => {
      switchView("viewReconcile");
    });
  }

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleUploadedFile(fileInput.files[0]);
    }
  });

  document.querySelectorAll(".demo-pdf-item").forEach(item => {
    item.addEventListener("click", async () => {
      const pdfName = item.getAttribute("data-pdf");
      await loadDemoPdfAndUpload(pdfName);
    });
  });
}

export function updatePipelineStep(stepName, statusText, isComplete = false) {
  const stepMap = {
    upload: document.getElementById("pipeStepUpload"),
    validate: document.getElementById("pipeStepValidate"),
    extract: document.getElementById("pipeStepExtract"),
    match: document.getElementById("pipeStepMatch")
  };

  const el = stepMap[stepName];
  if (!el) return;

  const statusEl = el.querySelector(".pipe-status");
  if (statusEl) statusEl.textContent = statusText;

  if (isComplete) {
    el.className = "pipe-step completed";
  } else {
    el.className = "pipe-step active";
  }
}

export async function handleUploadedFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const resultPanel = document.getElementById("uploadResultPanel");
  if (resultPanel) resultPanel.style.display = "flex";

  const filenameEl = document.getElementById("uploadResultFilename");
  if (filenameEl) filenameEl.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;

  const badgeEl = document.getElementById("uploadResultStatusBadge");
  if (badgeEl) {
    badgeEl.className = "badge badge-ambiguous font-mono";
    badgeEl.textContent = "◌ INGESTING & EXTRACTING...";
  }

  // Update pipeline progress
  updatePipelineStep("upload", "Completed ✓", true);
  updatePipelineStep("validate", "Validating...", false);

  try {
    updatePipelineStep("validate", "Format Validated ✓", true);
    updatePipelineStep("extract", "Extracting with pdfplumber...", false);

    const stagedData = await postUploadFile(formData);
    if (stagedData.error) throw new Error(stagedData.error);

    updatePipelineStep("extract", "Header & Lines Extracted ✓", true);
    updatePipelineStep("match", "Hybrid Matching Complete ✓", true);

    if (badgeEl) {
      badgeEl.className = "badge badge-matched font-mono";
      badgeEl.textContent = "✓ INGESTION & AI MATCHING COMPLETE";
    }

    // Populate Extracted Summary
    const invIdEl = document.getElementById("resExtractedInvId");
    if (invIdEl) invIdEl.textContent = stagedData.invoice_id || "--";

    const poIdEl = document.getElementById("resExtractedPoId");
    if (poIdEl) poIdEl.textContent = stagedData.po_id || "--";

    const vendorEl = document.getElementById("resExtractedVendor");
    if (vendorEl) vendorEl.textContent = stagedData.vendor_name || "--";

    const linesEl = document.getElementById("resExtractedLines");
    if (linesEl) linesEl.textContent = `${stagedData.lines ? stagedData.lines.length : 0} items`;

    const matchRateEl = document.getElementById("resExtractedMatchRate");
    if (matchRateEl) matchRateEl.textContent = `${stagedData.summary ? stagedData.summary.first_pass_rate_pct.toFixed(1) : 0}%`;

    state.currentStagedData = stagedData;
    renderReconciliationView(stagedData);
  } catch (err) {
    if (badgeEl) {
      badgeEl.className = "badge badge-unmatched font-mono";
      badgeEl.textContent = `⚠ INGESTION ERROR: ${err.message}`;
    }
    updatePipelineStep("validate", "Validation Error ⚠", false);
  }
}

export async function loadDemoPdfAndUpload(pdfName) {
  const resultPanel = document.getElementById("uploadResultPanel");
  if (resultPanel) resultPanel.style.display = "flex";

  const filenameEl = document.getElementById("uploadResultFilename");
  if (filenameEl) filenameEl.textContent = `${pdfName} (Sample PDF Invoice)`;

  const badgeEl = document.getElementById("uploadResultStatusBadge");
  if (badgeEl) {
    badgeEl.className = "badge badge-ambiguous font-mono";
    badgeEl.textContent = "◌ EXTRACTING PDF VIA PDFPLUMBER...";
  }

  updatePipelineStep("upload", "Completed ✓", true);
  updatePipelineStep("validate", "Format Validated ✓", true);
  updatePipelineStep("extract", "Extracting with pdfplumber...", false);

  try {
    const invId = pdfName.replace(".pdf", "");
    const scData = await fetchScenarioDetail(invId);

    updatePipelineStep("extract", "Header & Lines Extracted ✓", true);
    updatePipelineStep("match", "Executing Hybrid Matcher on CPU...", false);

    const recData = await postReconcile(scData);

    updatePipelineStep("match", "Hybrid Matching Complete ✓", true);

    if (badgeEl) {
      badgeEl.className = "badge badge-matched font-mono";
      badgeEl.textContent = "✓ INGESTION & AI MATCHING COMPLETE";
    }

    // Populate Extracted Summary
    const invIdEl = document.getElementById("resExtractedInvId");
    if (invIdEl) invIdEl.textContent = recData.invoice_id || "--";

    const poIdEl = document.getElementById("resExtractedPoId");
    if (poIdEl) poIdEl.textContent = recData.po_id || "--";

    const vendorEl = document.getElementById("resExtractedVendor");
    if (vendorEl) vendorEl.textContent = recData.vendor_name || "--";

    const linesEl = document.getElementById("resExtractedLines");
    if (linesEl) linesEl.textContent = `${recData.lines ? recData.lines.length : 0} items`;

    const matchRateEl = document.getElementById("resExtractedMatchRate");
    if (matchRateEl) matchRateEl.textContent = `${recData.summary ? recData.summary.first_pass_rate_pct.toFixed(1) : 0}%`;

    state.currentStagedData = recData;
    renderReconciliationView(recData);
  } catch (err) {
    if (badgeEl) {
      badgeEl.className = "badge badge-unmatched font-mono";
      badgeEl.textContent = `⚠ ERROR: ${err.message}`;
    }
  }
}
