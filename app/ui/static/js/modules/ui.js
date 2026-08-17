/**
 * SmartReconcile AI — Shared UI, Formatting & Category Helpers
 */

export function formatNumber(num) {
  if (num === null || num === undefined) return "0";
  return Math.round(num).toLocaleString("en-US");
}

export function formatFlagLabel(flag) {
  if (!flag) return "MATCH";
  return flag.replace("FLAG_", "").replace(/_/g, " ");
}

export function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

export function getCategoryIcon(cat) {
  const map = {
    standard_clean: "🌟",
    abbreviation_variation: "📝",
    spec_distractor: "🔍",
    uom_discrepancy: "⚖️",
    price_variance: "💰",
    quantity_variance: "📦",
    math_error: "🧮",
    unmatched_invoice_line: "🚫",
    partial_delivery: "📋"
  };
  return map[cat] || "📄";
}

export function formatCategoryLabel(cat) {
  if (!cat) return "";
  return cat.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}
