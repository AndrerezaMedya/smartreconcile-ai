"""
Deterministic 4-Way Commercial Verification Engine.
Checks: Quantity, Unit Price, Arithmetic Math, and Unit of Measure (UOM).
"""

from typing import Optional, Tuple
from app.core.config import PRICE_TOLERANCE_PCT, MATH_TOLERANCE_IDR
from app.core.models import InvoiceLine, POLine, VerificationFlags


class NumericVerifier:
    """
    Evaluates commercial consistency between an invoice line and an assigned PO line.
    """

    @staticmethod
    def normalize_uom(uom: str) -> str:
        """Standardizes common Indonesian and English UOM tokens."""
        u = uom.lower().strip()
        mapping = {
            "pcs": "pcs", "pc": "pcs", "piece": "pcs", "pieces": "pcs", "buah": "pcs", "biji": "pcs",
            "mtr": "meter", "m": "meter", "meter": "meter",
            "btg": "batang", "batang": "batang", "lonjor": "batang", "joint": "batang",
            "roll": "roll", "rol": "roll",
            "drum": "drum", "drm": "drum",
            "pail": "pail", "ember": "pail",
            "dus": "dus", "box": "dus", "kotak": "dus", "pak": "dus", "pack": "dus",
            "lbr": "lembar", "lembar": "lembar", "sheet": "lembar",
            "kg": "kg", "kilogram": "kg",
            "psg": "pasang", "pasang": "pasang", "pair": "pasang", "pairs": "pasang",
            "lsn": "lusin", "lusin": "lusin", "dozen": "lusin", "doz": "lusin",
            "set": "set"
        }
        return mapping.get(u, u)

    @classmethod
    def verify(
        cls,
        inv_line: InvoiceLine,
        po_line: Optional[POLine]
    ) -> VerificationFlags:
        """
        Runs complete 4-way verification check against the assigned PO line.
        """
        flags = VerificationFlags()
        details = []

        # 1. Line Arithmetic Math Check (Self-contained within invoice line)
        calc_total = inv_line.qty * inv_line.unit_price
        math_diff = abs(inv_line.line_total - calc_total)
        if math_diff > MATH_TOLERANCE_IDR:
            flags.math_flag = "FLAG_MATH_ERROR"
            flags.math_diff = math_diff
            flags.has_discrepancy = True
            details.append(
                f"Arithmetic Error: Invoiced total Rp {inv_line.line_total:,.0f} != "
                f"Calculated Rp {calc_total:,.0f} (Diff: Rp {math_diff:,.0f})"
            )

        if po_line is None:
            # Unmatched line - no PO to verify against
            flags.has_discrepancy = True
            details.append("Unmatched line: No corresponding PO line found in procurement order.")
            flags.discrepancy_details = details
            return flags

        # 2. Quantity Check
        qty_diff = inv_line.qty - po_line.ordered_qty
        flags.qty_diff = qty_diff
        if inv_line.qty > po_line.ordered_qty:
            flags.qty_flag = "FLAG_QTY_OVERBILLING"
            flags.has_discrepancy = True
            details.append(
                f"Quantity Overbilling: Invoiced {inv_line.qty:g} {inv_line.uom} > "
                f"PO ordered {po_line.ordered_qty:g} {po_line.uom} (+{qty_diff:g})"
            )
        elif inv_line.qty < po_line.ordered_qty:
            flags.qty_flag = "FLAG_QTY_UNDERBILLING"
            # Underbilling / partial shipment is noted but not strictly blocking if partial delivery is allowed
            details.append(
                f"Partial Shipment: Invoiced {inv_line.qty:g} < "
                f"PO ordered {po_line.ordered_qty:g} (Remaining: {-qty_diff:g})"
            )

        # 3. Unit Price Check
        if po_line.unit_price > 0:
            price_diff_pct = abs(inv_line.unit_price - po_line.unit_price) / po_line.unit_price
            flags.price_diff_pct = price_diff_pct
            if price_diff_pct > PRICE_TOLERANCE_PCT:
                flags.price_flag = "FLAG_PRICE_VARIANCE"
                flags.has_discrepancy = True
                pct_str = f"{price_diff_pct * 100:.1f}%"
                direction = "higher" if inv_line.unit_price > po_line.unit_price else "lower"
                details.append(
                    f"Unit Price Discrepancy: Invoiced Rp {inv_line.unit_price:,.0f} is {pct_str} {direction} "
                    f"than PO rate Rp {po_line.unit_price:,.0f} (Tolerance: {PRICE_TOLERANCE_PCT*100:.0f}%)"
                )

        # 4. UOM Compatibility Check
        inv_uom_norm = cls.normalize_uom(inv_line.uom)
        po_uom_norm = cls.normalize_uom(po_line.uom)
        if inv_uom_norm != po_uom_norm:
            flags.uom_flag = "FLAG_UOM_MISMATCH"
            flags.has_discrepancy = True
            details.append(
                f"UOM Incompatibility: Invoiced unit '{inv_line.uom}' does not match PO unit '{po_line.uom}'."
            )

        flags.discrepancy_details = details
        return flags
