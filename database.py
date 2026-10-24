"""
database.py
Thin data-access layer over SQLite. All queries are parameterised
(? placeholders), so user input is never pasted into SQL.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import config
from config import AGENT_ROUTING, DEFAULT_AGENT, SLA_HOURS

# Tests point this at a temporary file.
DB_PATH = config.DB_PATH


def _now():
    """Current UTC time as a naive datetime (stored as ISO text)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(reset: bool = False):
    if reset and DB_PATH.exists():
        DB_PATH.unlink()

    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT NOT NULL,
            customer_email TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            category_confidence REAL NOT NULL,
            priority TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'New',
            assigned_to TEXT,
            resolution_notes TEXT,
            created_at TEXT NOT NULL,
            due_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_complaints_due ON complaints(due_at)"
    )
    conn.commit()
    conn.close()


def create_complaint(customer_name, customer_email, description, category,
                      category_confidence, priority, sentiment):
    now = _now()
    due_at = now + timedelta(hours=SLA_HOURS.get(priority, 72))
    assigned_to = AGENT_ROUTING.get(category, DEFAULT_AGENT)

    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO complaints
            (customer_name, customer_email, description, category,
             category_confidence, priority, sentiment, status,
             assigned_to, created_at, due_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'New', ?, ?, ?, ?)
        """,
        (customer_name, customer_email, description, category,
         category_confidence, priority, sentiment, assigned_to,
         now.isoformat(timespec="seconds"),
         due_at.isoformat(timespec="seconds"),
         now.isoformat(timespec="seconds")),
    )
    conn.commit()
    complaint_id = cur.lastrowid
    conn.close()
    return complaint_id


def list_complaints(status=None, category=None, priority=None):
    query = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if category:
        query += " AND category = ?"
        params.append(category)
    if priority:
        query += " AND priority = ?"
        params.append(priority)
    query += " ORDER BY due_at ASC"

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_complaint(complaint_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM complaints WHERE id = ?", (complaint_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_complaint(complaint_id, status=None, priority=None,
                     assigned_to=None, resolution_notes=None):
    """Update a complaint. Changing priority also recalculates the SLA
    deadline, counted from when the complaint was created."""
    existing = get_complaint(complaint_id)
    if existing is None:
        return False

    now_iso = _now().isoformat(timespec="seconds")
    fields, params = [], []
    if status:
        fields.append("status = ?")
        params.append(status)
    if priority:
        fields.append("priority = ?")
        params.append(priority)
        created = datetime.fromisoformat(existing["created_at"])
        new_due = created + timedelta(hours=SLA_HOURS.get(priority, 72))
        fields.append("due_at = ?")
        params.append(new_due.isoformat(timespec="seconds"))
    if assigned_to:
        fields.append("assigned_to = ?")
        params.append(assigned_to)
    if resolution_notes is not None:
        fields.append("resolution_notes = ?")
        params.append(resolution_notes)

    fields.append("updated_at = ?")
    params.append(now_iso)

    if status == "Resolved":
        fields.append("resolved_at = ?")
        params.append(now_iso)

    params.append(complaint_id)
    conn = get_db()
    conn.execute(
        f"UPDATE complaints SET {', '.join(fields)} WHERE id = ?", params
    )
    conn.commit()
    conn.close()
    return True


def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]

    by_status = conn.execute(
        "SELECT status, COUNT(*) c FROM complaints GROUP BY status"
    ).fetchall()
    by_category = conn.execute(
        "SELECT category, COUNT(*) c FROM complaints GROUP BY category"
    ).fetchall()
    by_priority = conn.execute(
        "SELECT priority, COUNT(*) c FROM complaints GROUP BY priority"
    ).fetchall()

    now_iso = _now().isoformat(timespec="seconds")
    overdue = conn.execute(
        """
        SELECT COUNT(*) c FROM complaints
        WHERE due_at < ? AND status NOT IN ('Resolved', 'Closed')
        """,
        (now_iso,),
    ).fetchone()["c"]

    conn.close()
    return {
        "total": total,
        "overdue": overdue,
        "by_status": {r["status"]: r["c"] for r in by_status},
        "by_category": {r["category"]: r["c"] for r in by_category},
        "by_priority": {r["priority"]: r["c"] for r in by_priority},
    }
