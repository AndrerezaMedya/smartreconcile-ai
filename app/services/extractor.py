"""
Data Extraction Engine for SmartReconcile AI.
Parses Invoice Headers and Line Items from PDF, CSV, and JSON invoice files.
"""

import io
import re
import csv
import json
from typing import Dict, Any, Tuple, Optional, List
import pdfplumber

from app.core.models import Invoice, InvoiceLine, PurchaseOrder, POLine


class InvoiceExtractor:

    @classmethod
    def extract(cls, filename: str, content_bytes: bytes) -> Tuple[Invoice, Optional[str]]:
        """
        Main extraction entrypoint.
        Returns: (Invoice domain object, extracted_po_id_reference)
        """
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == ".json":
            return cls.extract_json(content_bytes)
        elif ext == ".csv":
            return cls.extract_csv(content_bytes)
        elif ext == ".pdf":
            return cls.extract_pdf(content_bytes)
        else:
            raise ValueError(f"Unsupported file format for extraction: '{ext}'")

    @classmethod
    def extract_json(cls, content_bytes: bytes) -> Tuple[Invoice, Optional[str]]:
        """Parses structured JSON invoice payload."""
        data = json.loads(content_bytes.decode("utf-8"))

        # Check if structured as {"invoice": {...}, "purchase_order": {...}}
        inv_data = data.get("invoice", data)
        po_id_ref = inv_data.get("po_id") or data.get("po_id")

        lines = [
            InvoiceLine(
                line_no=l["line_no"],
                description=l["description"],
                qty=float(l["qty"]),
                uom=l["uom"],
                unit_price=float(l["unit_price"]),
                line_total=float(l.get("line_total", float(l["qty"]) * float(l["unit_price"])))
            )
            for l in inv_data.get("invoice_lines", [])
        ]

        invoice = Invoice(
            invoice_id=inv_data.get("invoice_id", "INV-UNKNOWN"),
            po_id=po_id_ref or "PO-UNKNOWN",
            vendor_name=inv_data.get("vendor_name", "Unknown Vendor"),
            invoice_date=inv_data.get("invoice_date", "2026-08-14"),
            scenario_category=inv_data.get("scenario_category", "uploaded_custom"),
            scenario_description=inv_data.get("scenario_description", "Uploaded invoice document"),
            invoice_lines=lines
        )
        return invoice, po_id_ref

    @classmethod
    def extract_csv(cls, content_bytes: bytes) -> Tuple[Invoice, Optional[str]]:
        """Parses CSV invoice with optional metadata header comments."""
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1")

        lines = text.strip().splitlines()
        invoice_id = "INV-UPLOADED"
        po_id = None
        vendor_name = "Uploaded Vendor"
        invoice_date = "2026-08-14"

        data_rows = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check for metadata comments (e.g. # INVOICE: INV-2026-001)
            if trimmed.startswith("#") or ":" in trimmed and not trimmed[0].isdigit():
                upper = trimmed.upper()
                if "INVOICE" in upper:
                    m = re.search(r"(INV-[\w\-]+)", trimmed, re.I)
                    if m: invoice_id = m.group(1).upper()
                if "PO" in upper or "PURCHASE ORDER" in upper:
                    m = re.search(r"(PO-[\w\-]+)", trimmed, re.I)
                    if m: po_id = m.group(1).upper()
                if "VENDOR" in upper:
                    parts = trimmed.split(":", 1)
                    if len(parts) > 1: vendor_name = parts[1].strip()
                if "DATE" in upper:
                    m = re.search(r"(\d{4}-\d{2}-\d{2})", trimmed)
                    if m: invoice_date = m.group(1)
                continue

            data_rows.append(trimmed)

        # Parse CSV table
        reader = csv.reader(data_rows)
        inv_lines = []
        header_skipped = False

        for row in reader:
            if not row or len(row) < 5:
                continue

            # Check if header row
            if not header_skipped and any("desc" in col.lower() or "line" in col.lower() for col in row):
                header_skipped = True
                continue

            try:
                line_no = int(row[0].strip()) if row[0].strip().isdigit() else len(inv_lines) + 1
                desc = row[1].strip()
                qty = float(row[2].strip().replace(",", ""))
                uom = row[3].strip()
                price = float(row[4].strip().replace(",", ""))
                total = float(row[5].strip().replace(",", "")) if len(row) > 5 else qty * price

                inv_lines.append(InvoiceLine(
                    line_no=line_no,
                    description=desc,
                    qty=qty,
                    uom=uom,
                    unit_price=price,
                    line_total=total
                ))
            except Exception:
                continue

        invoice = Invoice(
            invoice_id=invoice_id,
            po_id=po_id or "PO-UNKNOWN",
            vendor_name=vendor_name,
            invoice_date=invoice_date,
            scenario_category="uploaded_csv",
            scenario_description=f"CSV Upload with {len(inv_lines)} line items",
            invoice_lines=inv_lines
        )
        return invoice, po_id

    @classmethod
    def extract_pdf(cls, content_bytes: bytes) -> Tuple[Invoice, Optional[str]]:
        """
        Parses text and tabular lines from digital PDF invoice via pdfplumber.
        """
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

        if not full_text.strip():
            raise ValueError("PDF does not contain extractable digital text.")

        # Extract Header Heuristics
        invoice_id = "INV-UPLOADED"
        m_inv_code = re.search(r"(INV-[\w\-]+)", full_text, re.I)
        if m_inv_code:
            invoice_id = m_inv_code.group(1).upper().strip()
        else:
            m_inv = re.search(r"INVOICE\s*(?:NUMBER|NO|#)\s*[:=]\s*([^\s\n]+)", full_text, re.I)
            if m_inv:
                invoice_id = m_inv.group(1).strip()

        po_id = None
        m_po_code = re.search(r"(PO-[\w\-]+)", full_text, re.I)
        if m_po_code:
            po_id = m_po_code.group(1).upper().strip()
        else:
            m_po = re.search(r"(?:PO|PURCHASE\s*ORDER)\s*(?:REFERENCE|REF|NO|#)\s*[:=]\s*([^\s\n]+)", full_text, re.I)
            if m_po:
                po_id = m_po.group(1).strip()

        vendor_name = "Extracted Vendor"
        m_ven = re.search(r"VENDOR\s*[:=]?\s*([^\n\r]+)", full_text, re.I)
        if m_ven:
            vendor_name = m_ven.group(1).strip()

        invoice_date = "2026-08-14"
        m_date = re.search(r"DATE\s*[:=]?\s*(\d{4}-\d{2}-\d{2})", full_text, re.I)
        if m_date:
            invoice_date = m_date.group(1).strip()

        # Parse Line Items (matches rows like: 1 | Pipa Galvanis 2 inch Sch 40 | 50.0 | btg | 450000 | 22500000)
        inv_lines = []
        raw_lines = full_text.splitlines()

        for rline in raw_lines:
            rline_str = rline.strip()
            if "|" in rline_str:
                parts = [p.strip() for p in rline_str.split("|")]
                if len(parts) >= 5 and (parts[0].isdigit() or re.match(r"^\d+$", parts[0])):
                    try:
                        line_no = int(parts[0])
                        desc = parts[1]
                        qty = float(parts[2].replace(",", ""))
                        uom = parts[3]
                        price = float(parts[4].replace(",", "").replace("Rp", "").strip())
                        total = float(parts[5].replace(",", "").replace("Rp", "").strip()) if len(parts) > 5 else qty * price

                        inv_lines.append(InvoiceLine(
                            line_no=line_no,
                            description=desc,
                            qty=qty,
                            uom=uom,
                            unit_price=price,
                            line_total=total
                        ))
                    except Exception:
                        continue

        if not inv_lines:
            # Fallback regex line parser for space/tab delimited rows
            for rline in raw_lines:
                m_row = re.match(r"^(\d+)\s+([A-Za-z0-9\s\"\'\.\-\/\(\)]+?)\s+(\d+(?:\.\d+)?)\s+([a-zA-Z]+)\s+([\d\,\.]+)\s+([\d\,\.]+)", rline.strip())
                if m_row:
                    try:
                        l_no = int(m_row.group(1))
                        l_desc = m_row.group(2).strip()
                        l_qty = float(m_row.group(3))
                        l_uom = m_row.group(4).strip()
                        l_price = float(m_row.group(5).replace(",", ""))
                        l_total = float(m_row.group(6).replace(",", ""))
                        inv_lines.append(InvoiceLine(
                            line_no=l_no,
                            description=l_desc,
                            qty=l_qty,
                            uom=l_uom,
                            unit_price=l_price,
                            line_total=l_total
                        ))
                    except Exception:
                        continue

        invoice = Invoice(
            invoice_id=invoice_id,
            po_id=po_id or "PO-UNKNOWN",
            vendor_name=vendor_name,
            invoice_date=invoice_date,
            scenario_category="uploaded_pdf",
            scenario_description=f"PDF Document with {len(inv_lines)} extracted lines",
            invoice_lines=inv_lines
        )
        return invoice, po_id
