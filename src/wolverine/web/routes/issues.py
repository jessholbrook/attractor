from __future__ import annotations

from flask import Blueprint, abort, current_app, render_template, request

from wolverine.model.issue import IssueStatus

issues_bp = Blueprint("issues", __name__)


@issues_bp.route("/")
def list_issues():
    """List all issues, optionally filtered by status."""
    issue_repo = current_app.extensions["issue_repo"]
    status_filter = request.args.get("status")

    if status_filter:
        issues = issue_repo.list_by_status(IssueStatus(status_filter))
    else:
        issues = issue_repo.list_all()

    return render_template(
        "issues/list.html",
        issues=issues,
        status_filter=status_filter,
    )


@issues_bp.route("/<issue_id>")
def detail(issue_id: str):
    """Show issue detail with related signals and solutions."""
    issue_repo = current_app.extensions["issue_repo"]
    solution_repo = current_app.extensions["solution_repo"]
    signal_repo = current_app.extensions["signal_repo"]

    issue = issue_repo.get(issue_id)
    if issue is None:
        abort(404)

    solutions = solution_repo.list_by_issue(issue_id)
    signals = tuple(
        signal_repo.get(sid)
        for sid in issue.signal_ids
        if signal_repo.get(sid) is not None
    )

    return render_template(
        "issues/detail.html",
        issue=issue,
        solutions=solutions,
        signals=signals,
    )
