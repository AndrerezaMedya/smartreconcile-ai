/**
 * SmartReconcile AI — Centralized REST API Service
 */

export async function fetchScenarios() {
  const res = await fetch("/api/scenarios");
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function fetchScenarioDetail(invoiceId) {
  const res = await fetch(`/api/scenarios/${encodeURIComponent(invoiceId)}`);
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function postReconcile(invoiceData) {
  const res = await fetch("/api/reconcile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(invoiceData)
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function postUploadFile(formData) {
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function postReviewAction(payload) {
  const res = await fetch("/api/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function postCommit(stagingId, payload) {
  const res = await fetch(`/api/staging/${encodeURIComponent(stagingId)}/commit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function fetchCommittedLedger() {
  const res = await fetch("/api/committed");
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function fetchAuditLog() {
  const res = await fetch("/api/audit-log");
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}

export async function postBenchmarkRunAll() {
  const res = await fetch("/api/benchmark/run-all", {
    method: "POST"
  });
  if (!res.ok) throw new Error(`HTTP error ${res.status}`);
  return await res.json();
}
