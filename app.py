"""
app.py
Flask entry point for the AI-assisted complaint management system.

  Public routes : /submit, /track/<id>, /api/classify-preview, /api/model-info
  Staff routes  : /login, /console, /complaint/<id>, /api/complaints, /api/stats
                  (all staff routes require login)
"""

import logging
import re
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import config
import database
from ai_classifier import classify_complaint, model_info

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    CSRF_ENABLED=True,
    MAX_CONTENT_LENGTH=64 * 1024,  # reject oversized requests
)

_PASSWORD_HASH = generate_password_hash(config.STAFF_PASSWORD)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

STATUSES = config.STATUSES
PRIORITIES = config.PRIORITIES

if config.USING_DEFAULT_CREDENTIALS:
    log.warning("Using default staff credentials (admin / admin123). "
                "Set STAFF_USERNAME and STAFF_PASSWORD for any shared use.")


# ---------------------------------------------------------------------------
# Security helpers: CSRF protection and login
# ---------------------------------------------------------------------------

def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def csrf_protect():
    if not app.config["CSRF_ENABLED"] or request.method != "POST":
        return
    if request.endpoint == "classify_preview":  # JSON, changes no state
        return
    sent = request.form.get("_csrf", "")
    if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
        abort(400, "Invalid or missing CSRF token")


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("staff"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "authentication required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def _safe_next(target):
    """Only allow redirects to paths on this site."""
    if target and target.startswith("/") and not urlparse(target).netloc \
            and not target.startswith("//"):
        return target
    return url_for("console")


@app.context_processor
def inject_model_info():
    return {"model": model_info(), "is_staff": bool(session.get("staff"))}


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = request.form.get("username", "")
        pw = request.form.get("password", "")
        user_ok = secrets.compare_digest(user, config.STAFF_USERNAME)
        pw_ok = check_password_hash(_PASSWORD_HASH, pw)
        if user_ok and pw_ok:
            session.clear()
            session["staff"] = True
            return redirect(_safe_next(request.args.get("next")))
        error = "Incorrect username or password."
    return render_template("login.html", error=error), (401 if error else 200)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Public: customer complaint intake
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return redirect(url_for("submit"))


@app.route("/submit", methods=["GET", "POST"])
def submit():
    error = None
    form = {"name": "", "email": "", "description": ""}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip()
                for k in ("name", "email", "description")}
        if not (2 <= len(form["name"]) <= 100):
            error = "Please enter your name (2–100 characters)."
        elif not _EMAIL_RE.match(form["email"]) or len(form["email"]) > 254:
            error = "Please enter a valid email address."
        elif not (15 <= len(form["description"]) <= 5000):
            error = "Please describe the problem in 15–5000 characters."

        if error is None:
            result = classify_complaint(form["description"])
            complaint_id = database.create_complaint(
                customer_name=form["name"],
                customer_email=form["email"],
                description=form["description"],
                category=result["category"],
                category_confidence=result["category_confidence"],
                priority=result["priority"],
                sentiment=result["sentiment"],
            )
            return redirect(url_for("track", complaint_id=complaint_id))
    return render_template("submit.html", error=error, form=form), \
        (400 if error else 200)


@app.route("/track/<int:complaint_id>")
def track(complaint_id):
    complaint = database.get_complaint(complaint_id)
    if not complaint:
        return render_template("not_found.html"), 404
    return render_template("track.html", c=complaint)


@app.route("/api/classify-preview", methods=["POST"])
def classify_preview():
    """Live preview used by the submit form while the customer types."""
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("description", ""))[:5000]
    if len(text.strip()) < 12:
        return jsonify({"ready": False})
    result = classify_complaint(text)
    result["ready"] = True
    return jsonify(result)


@app.route("/api/model-info")
def api_model_info():
    return jsonify(model_info())


# ---------------------------------------------------------------------------
# Staff: triage console (login required)
# ---------------------------------------------------------------------------

@app.route("/console")
@login_required
def console():
    status = request.args.get("status") or None
    category = request.args.get("category") or None
    priority = request.args.get("priority") or None

    return render_template(
        "console.html",
        complaints=database.list_complaints(status, category, priority),
        stats=database.get_stats(),
        categories=config.CATEGORIES,
        statuses=STATUSES,
        priorities=PRIORITIES,
        active_status=status,
        active_category=category,
        active_priority=priority,
        now=database._now().isoformat(timespec="seconds"),
    )


@app.route("/complaint/<int:complaint_id>")
@login_required
def complaint_detail(complaint_id):
    complaint = database.get_complaint(complaint_id)
    if not complaint:
        return render_template("not_found.html"), 404
    return render_template("complaint_detail.html", c=complaint,
                           statuses=STATUSES, priorities=PRIORITIES)


@app.route("/complaint/<int:complaint_id>/update", methods=["POST"])
@login_required
def update_complaint(complaint_id):
    status = request.form.get("status")
    priority = request.form.get("priority")
    if status not in STATUSES or priority not in PRIORITIES:
        abort(400, "Invalid status or priority")
    notes = request.form.get("resolution_notes", "")[:2000]
    if not database.update_complaint(complaint_id, status=status,
                                     priority=priority, resolution_notes=notes):
        return render_template("not_found.html"), 404
    return redirect(url_for("complaint_detail", complaint_id=complaint_id))


# ---------------------------------------------------------------------------
# JSON API (staff only: it exposes customer data)
# ---------------------------------------------------------------------------

@app.route("/api/complaints")
@login_required
def api_complaints():
    return jsonify(database.list_complaints(
        status=request.args.get("status"),
        category=request.args.get("category"),
        priority=request.args.get("priority"),
    ))


@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(database.get_stats())


database.init_db()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
