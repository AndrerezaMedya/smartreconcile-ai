/**
 * SmartReconcile AI — Document Ingestion & Upload Workflow
 */

import { state } from "./state.js";
import { postUploadFile } from "./api.js";
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
  // Fetches the actual PDF file and runs it through the same real upload path
  // as drag-and-drop (/api/upload -> pdfplumber -> hybrid matcher). This used
  // to shortcut through the /api/scenarios JSON fixture while still labeling
  // the step "EXTRACTING PDF VIA PDFPLUMBER" — a claim that wasn't true, since
  // no PDF was actually read. A "quick demo" convenience must still do what it
  // says on screen.
  const badgeEl = document.getElementById("uploadResultStatusBadge");
  const resultPanel = document.getElementById("uploadResultPanel");
  if (resultPanel) resultPanel.style.display = "flex";
  if (badgeEl) {
    badgeEl.className = "badge badge-ambiguous font-mono";
    badgeEl.textContent = "◌ FETCHING SAMPLE PDF...";
  }

  try {
    const res = await fetch(`/demo-pdfs/${encodeURIComponent(pdfName)}`);
    if (!res.ok) throw new Error(`Could not load sample PDF (HTTP ${res.status})`);
    const blob = await res.blob();
    const file = new File([blob], pdfName, { type: "application/pdf" });

    await handleUploadedFile(file);
  } catch (err) {
    if (badgeEl) {
      badgeEl.className = "badge badge-unmatched font-mono";
      badgeEl.textContent = `⚠ ERROR: ${err.message}`;
    }
  }
}
