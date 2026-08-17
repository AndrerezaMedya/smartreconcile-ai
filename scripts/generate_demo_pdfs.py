"""
Generates realistic digital PDF invoices for demo scenarios in data/demo_invoices_pdf/
Uses a pure Python standard PDF 1.4 generator readable by any PDF viewer and pdfplumber.
"""

import json
from pathlib import Path
from app.core.config import DATA_DIR

PDF_OUT_DIR = DATA_DIR / "demo_invoices_pdf"
PDF_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_INVOICES_PATH = DATA_DIR / "workflow_simulation_invoices.json"


def generate_pdf_bytes(title_lines, table_lines):
    """Generates standard PDF 1.4 binary content."""
    stream = "BT\n/F1 10 Tf\n15 TL\n50 780 Td\n"
    for line in title_lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream += f"({escaped}) '\n"
    
    stream += "() '\n"
    stream += "(LINE ITEMS:) '\n"
    stream += "(# | Description | Qty | UOM | Unit Price (Rp) | Total (Rp)) '\n"
    stream += "(-------------------------------------------------------------------------------------) '\n"
    
    for line in table_lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream += f"({escaped}) '\n"
        
    stream += "ET\n"
    s_bytes = stream.encode("latin1")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []

    offsets.append(len(pdf))
    pdf.extend(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    offsets.append(len(pdf))
    pdf.extend(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

    offsets.append(len(pdf))
    pdf.extend(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")

    offsets.append(len(pdf))
    pdf.extend(f"4 0 obj\n<< /Length {len(s_bytes)} >>\nstream\n".encode("latin1"))
    pdf.extend(s_bytes)
    pdf.extend(b"endstream\nendobj\n")

    offsets.append(len(pdf))
    pdf.extend(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 6\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin1"))

    pdf.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1"))
    return bytes(pdf)


def main():
    if not SIM_INVOICES_PATH.exists():
        print(f"Simulation dataset not found: {SIM_INVOICES_PATH}")
        return

    with open(SIM_INVOICES_PATH, encoding="utf-8") as f:
        invoices = json.load(f)

    print(f"Generating PDF invoices for {len(invoices)} scenarios...")
    count = 0
    for inv in invoices:
        inv_id = inv["invoice_id"]
        po_id = inv["po_id"]
        vendor = inv["vendor_name"]
        date_str = inv["invoice_date"]
        category = inv.get("scenario_category", "Standard")

        title_lines = [
            f"COMMERCIAL TAX INVOICE",
            f"INVOICE NUMBER: {inv_id}",
            f"PO REFERENCE: {po_id}",
            f"VENDOR: {vendor}",
            f"DATE: {date_str}",
            f"CATEGORY: {category}"
        ]

        table_lines = []
        for l in inv["invoice_lines"]:
            table_lines.append(
                f"{l['line_no']} | {l['description']} | {l['qty']:.1f} | {l['uom']} | {l['unit_price']:,.0f} | {l['line_total']:,.0f}"
            )

        pdf_bytes = generate_pdf_bytes(title_lines, table_lines)
        out_file = PDF_OUT_DIR / f"{inv_id}.pdf"
        with open(out_file, "wb") as f:
            f.write(pdf_bytes)
        count += 1

    print(f"Successfully generated {count} demo PDF invoices in {PDF_OUT_DIR}")


if __name__ == "__main__":
    main()
