"""
Purchase Order Repository for querying Master PO records and candidate line items.
"""

from typing import Optional, List, Dict, Any
from app.core.models import PurchaseOrder, POLine
from app.db.database import get_db_connection


class PORepository:
    @staticmethod
    def get_po_by_id(po_id: str) -> Optional[PurchaseOrder]:
        """Fetches a PurchaseOrder and its line items by PO ID."""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM purchase_orders WHERE po_id = ?", (po_id,))
        po_row = cursor.fetchone()
        if not po_row:
            conn.close()
            return None

        cursor.execute("SELECT * FROM po_lines WHERE po_id = ? ORDER BY po_line_no ASC", (po_id,))
        line_rows = cursor.fetchall()
        conn.close()

        po_lines = [
            POLine(
                po_line_no=lr["po_line_no"],
                description=lr["description"],
                ordered_qty=lr["ordered_qty"],
                uom=lr["uom"],
                unit_price=lr["unit_price"],
                line_total=lr["ordered_qty"] * lr["unit_price"]
            )
            for lr in line_rows
        ]

        return PurchaseOrder(
            po_id=po_row["po_id"],
            vendor_name=po_row["vendor_name"],
            po_date=po_row["po_date"],
            po_lines=po_lines
        )

    @staticmethod
    def list_all_pos() -> List[Dict[str, Any]]:
        """Lists all master PO summaries."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT po.po_id, po.vendor_name, po.po_date, po.total_amount, po.status,
                   COUNT(pl.id) as line_count
            FROM purchase_orders po
            LEFT JOIN po_lines pl ON po.po_id = pl.po_id
            GROUP BY po.po_id
            ORDER BY po.po_id ASC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "po_id": r["po_id"],
                "vendor_name": r["vendor_name"],
                "po_date": r["po_date"],
                "total_amount": r["total_amount"],
                "status": r["status"],
                "line_count": r["line_count"]
            }
            for r in rows
        ]
