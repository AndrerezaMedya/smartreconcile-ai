"""
SQLite Database Connection and Schema Initialization for SmartReconcile AI.
Acts as a lightweight local staging ledger and PO master repository for the MVP.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional

from app.core.config import DATA_DIR

DB_PATH = DATA_DIR / "reconcile_erp.db"
SIM_INVOICES_PATH = DATA_DIR / "workflow_simulation_invoices.json"


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database():
    """Initializes SQLite schema and auto-seeds Master POs if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Purchase Orders Master
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS purchase_orders (
        po_id TEXT PRIMARY KEY,
        vendor_name TEXT NOT NULL,
        po_date TEXT,
        total_amount REAL,
        status TEXT DEFAULT 'OPEN'
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS po_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        po_id TEXT NOT NULL,
        po_line_no INTEGER NOT NULL,
        description TEXT NOT NULL,
        ordered_qty REAL NOT NULL,
        uom TEXT NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (po_id) REFERENCES purchase_orders(po_id) ON DELETE CASCADE
    );
    """)

    # 2. Staged Reconciliations (Drafts awaiting review & confirmation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS staged_reconciliations (
        staging_id TEXT PRIMARY KEY,
        invoice_id TEXT NOT NULL,
        po_id TEXT NOT NULL,
        vendor_name TEXT NOT NULL,
        invoice_date TEXT,
        file_name TEXT,
        status TEXT NOT NULL, -- STAGED_PENDING_REVIEW, COMMITTED, REJECTED
        first_pass_rate REAL,
        confidently_matched_count INTEGER,
        ambiguous_count INTEGER,
        unmatched_count INTEGER,
        discrepancies_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS staged_lines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        staging_id TEXT NOT NULL,
        line_no INTEGER NOT NULL,
        invoice_description TEXT NOT NULL,
        invoice_qty REAL NOT NULL,
        invoice_uom TEXT NOT NULL,
        invoice_unit_price REAL NOT NULL,
        invoice_line_total REAL NOT NULL,
        status TEXT NOT NULL, -- MATCHED, AMBIGUOUS, UNMATCHED
        assigned_po_line_no INTEGER,
        assigned_po_description TEXT,
        assigned_po_qty REAL,
        assigned_po_uom TEXT,
        assigned_po_unit_price REAL,
        score REAL,
        confidence_margin REAL,
        is_semantic_routed BOOLEAN,
        has_discrepancy BOOLEAN,
        qty_flag TEXT,
        price_flag TEXT,
        math_flag TEXT,
        uom_flag TEXT,
        discrepancy_details TEXT, -- JSON array
        top_candidates_json TEXT, -- JSON array
        is_reviewed BOOLEAN DEFAULT 0,
        review_action TEXT,
        review_notes TEXT,
        FOREIGN KEY (staging_id) REFERENCES staged_reconciliations(staging_id) ON DELETE CASCADE
    );
    """)

    # 3. Committed Ledger (Permanent audit history of approved transactions)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS committed_invoices (
        commit_id TEXT PRIMARY KEY,
        staging_id TEXT NOT NULL,
        invoice_id TEXT NOT NULL,
        po_id TEXT NOT NULL,
        vendor_name TEXT NOT NULL,
        total_invoiced_amount REAL,
        total_matched_amount REAL,
        reviewer_name TEXT NOT NULL,
        commit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT,
        FOREIGN KEY (staging_id) REFERENCES staged_reconciliations(staging_id)
    );
    """)

    # 4. Audit Log Trail
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        staging_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        action TEXT NOT NULL,
        line_no INTEGER,
        details TEXT NOT NULL,
        user TEXT NOT NULL
    );
    """)

    conn.commit()

    # Seed Master POs if table is empty
    cursor.execute("SELECT COUNT(*) as count FROM purchase_orders")
    count = cursor.fetchone()["count"]
    if count == 0 and SIM_INVOICES_PATH.exists():
        print("Seeding Purchase Orders Master from simulation dataset...")
        with open(SIM_INVOICES_PATH, encoding="utf-8") as f:
            invoices = json.load(f)

        seeded_pos = set()
        for inv in invoices:
            po_id = inv["po_id"]
            if po_id in seeded_pos:
                continue

            po_lines = inv["po_lines"]
            total_amt = sum(pl["ordered_qty"] * pl["unit_price"] for pl in po_lines)

            cursor.execute(
                "INSERT INTO purchase_orders (po_id, vendor_name, po_date, total_amount, status) VALUES (?, ?, ?, ?, ?)",
                (po_id, inv["vendor_name"], inv["invoice_date"], total_amt, "OPEN")
            )

            for pl in po_lines:
                cursor.execute(
                    "INSERT INTO po_lines (po_id, po_line_no, description, ordered_qty, uom, unit_price) VALUES (?, ?, ?, ?, ?, ?)",
                    (po_id, pl["po_line_no"], pl["description"], pl["ordered_qty"], pl["uom"], pl["unit_price"])
                )

            seeded_pos.add(po_id)

        conn.commit()
        print(f"Seeded {len(seeded_pos)} master Purchase Orders.")

    conn.close()


# Initialize on import
init_database()
