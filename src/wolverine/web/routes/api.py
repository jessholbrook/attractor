from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from wolverine.model.signal import RawSignal, SignalKind, SignalSource

api_bp = Blueprint("api", __name__)


@api_bp.after_request
def add_cors_headers(response):
    """Allow cross-origin requests to the API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@api_bp.route("/signals", methods=["OPTIONS"])
def signals_preflight():
    """Handle CORS preflight for signal creation."""
    return "", 204


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


@api_bp.route("/signals", methods=["POST"])
def create_signal():
    """Create a signal via JSON API."""
    data = request.get_json()
    if not data or "title" not in data or "body" not in data:
        return jsonify({"error": "title and body required"}), 400

    signal = RawSignal(
        id=_generate_id(),
        kind=SignalKind(data.get("kind", "manual")),
        source=SignalSource.API,
        title=data["title"],
        body=data["body"],
        received_at=datetime.now(timezone.utc).isoformat(),
        metadata=data.get("metadata", {}),
    )
    signal_repo = current_app.extensions["signal_repo"]
    signal_repo.create(signal)

    return jsonify({"id": signal.id, "status": "accepted"}), 202


@api_bp.route("/issues/<issue_id>/status")
def issue_status(issue_id: str):
    """Get the current status of an issue."""
    issue_repo = current_app.extensions["issue_repo"]
    issue = issue_repo.get(issue_id)
    if issue is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": issue.id, "status": issue.status})


@api_bp.route("/runs/<run_id>/status")
def run_status(run_id: str):
    """Get the current status of a healing run."""
    run_repo = current_app.extensions["run_repo"]
    run = run_repo.get(run_id)
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"id": run.id, "status": run.status})


@api_bp.route("/dashboard/stats")
def dashboard_stats():
    """Return issue and run counts for the dashboard."""
    issue_repo = current_app.extensions["issue_repo"]
    run_repo = current_app.extensions["run_repo"]
    return jsonify({
        "issues": issue_repo.count_by_status(),
        "runs": run_repo.count_by_status(),
    })
