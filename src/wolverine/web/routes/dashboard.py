from __future__ import annotations

from flask import Blueprint, current_app, render_template

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """Dashboard showing issue/run stats and recent activity."""
    issue_repo = current_app.extensions["issue_repo"]
    run_repo = current_app.extensions["run_repo"]

    issue_counts = issue_repo.count_by_status()
    run_counts = run_repo.count_by_status()
    recent_runs = run_repo.list_recent(limit=10)

    return render_template(
        "dashboard.html",
        issue_counts=issue_counts,
        run_counts=run_counts,
        recent_runs=recent_runs,
    )
