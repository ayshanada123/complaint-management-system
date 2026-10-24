"""
config.py
Central settings: paths, SLA rules, routing table, and staff credentials.
Values can be overridden with environment variables (see .env.example).
"""

import os
import secrets
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).parent

DB_PATH = Path(os.environ.get("CRM_DB_PATH", BASE_DIR / "crm.db"))
MODEL_PATH = BASE_DIR / "models" / "category_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_complaints.csv"
CLEAN_DATA_PATH = BASE_DIR / "data" / "complaints_clean.csv"

# Flask session signing key. If not set, a random one is used (sessions
# reset on every restart, which is fine for local development).
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

# Staff console login. Set real values in your environment for any
# shared use; the defaults are for local development only.
STAFF_USERNAME = os.environ.get("STAFF_USERNAME", "admin")
STAFF_PASSWORD = os.environ.get("STAFF_PASSWORD", "admin123")
USING_DEFAULT_CREDENTIALS = "STAFF_PASSWORD" not in os.environ

# Complaint categories (derived from the CFPB "Product" field).
CATEGORIES = [
    "Credit Reporting",
    "Debt Collection",
    "Mortgage",
    "Credit & Prepaid Cards",
    "Bank Accounts & Services",
    "Loans",
    "Money Transfers",
]

STATUSES = ["New", "In Progress", "Resolved", "Closed"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]

# Hours allowed to resolve a complaint, per priority (the SLA).
SLA_HOURS = {"Urgent": 4, "High": 24, "Medium": 72, "Low": 120}

# Category -> agent routing table.
AGENT_ROUTING = {
    "Credit Reporting": "Asha Nair",
    "Debt Collection": "Rahul Menon",
    "Mortgage": "Devika Pillai",
    "Credit & Prepaid Cards": "Arjun Das",
    "Bank Accounts & Services": "Meera Krishnan",
    "Loans": "Nikhil Varma",
    "Money Transfers": "Sana Fathima",
}
DEFAULT_AGENT = "General Queue"

# Below this model confidence the complaint is flagged for human review.
LOW_CONFIDENCE_THRESHOLD = 0.35
