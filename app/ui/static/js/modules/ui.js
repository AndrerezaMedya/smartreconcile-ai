/**
 * SmartReconcile AI — Shared UI, Formatting & Category Helpers
 */

export function formatNumber(num) {
  if (num === null || num === undefined) return "0";
  return Math.round(num).toLocaleString("en-US");
}

/**
 * Formats a verifier flag (e.g. FLAG_PRICE_VARIANCE) for display, dropping the
 * leading category word when the flag already repeats it — flags follow
 * "<CATEGORY>_<MATCH|DESCRIPTOR>", so a caller that already shows "PRICE" as
 * a header doesn't get "PRICE PRICE VARIANCE" restating itself.
 */
export function formatFlagValue(flag, category) {
  const fallback = category === "MATH" ? "CORRECT" : "MATCH";
  if (!flag) return fallback;
  const stripped = flag.replace("FLAG_", "");
  const withoutCategory = stripped.startsWith(category + "_") ? stripped.slice(category.length + 1) : stripped;
  return withoutCategory.replace(/_/g, " ");
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

/**
 * Computes the same unit-price variance shown in the review modal, so the
 * reconciliation table row can surface the at-risk amount (FR-18) without a
 * second click — a reviewer scanning the table should see the money, not just
 * a status word. Returns null when there's nothing to compare (no PO price),
 * the variance is within the commercial tolerance already enforced server
 * side (NumericVerifier.PRICE_TOLERANCE_PCT, mirrored here as 1%), or the UOM
 * itself doesn't match: a per-roll price vs. a per-meter price aren't the same
 * unit, so a raw percentage between them (e.g. "+9900%") is meaningless noise,
 * not a real discrepancy — that comparison needs a UOM conversion the backend
 * doesn't compute today, so we suppress it here rather than show a wrong number.
 */
export function getPriceVariance(line) {
  const uomFlag = line.verification && line.verification.uom_flag;
  if (uomFlag && uomFlag !== "UOM_MATCH") return null;

  const invPrice = line.invoice_unit_price || 0;
  const poPrice = line.assigned_po_unit_price || 0;
  if (!poPrice || Math.abs(invPrice - poPrice) <= poPrice * 0.01) return null;

  const pct = ((invPrice - poPrice) / poPrice) * 100;
  const invTotal = line.invoice_line_total || 0;
  const poTotal = (line.assigned_po_qty || 0) * poPrice;
  return { pct, unitDiff: invPrice - poPrice, totalDiff: invTotal - poTotal };
}
