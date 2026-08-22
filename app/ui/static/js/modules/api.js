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

export async function postStagingReviewAction(stagingId, payload) {
  /** DB-backed review: persists action to SQLite staging, captures audit trail. */
  const res = await fetch(`/api/staging/${encodeURIComponent(stagingId)}/review`, {
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

// NOTE: /api/benchmark/run-all is intentionally not exposed in the UI.
// The endpoint remains available for generating benchmark figures offline
// (see scripts/), but batch tooling is out of scope for the MVP surface.
