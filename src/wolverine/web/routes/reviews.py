from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask import Blueprint, abort, current_app, redirect, render_template, request, url_for

from wolverine.model.issue import IssueStatus
from wolverine.model.review import Review, ReviewDecision
from wolverine.model.solution import SolutionStatus

reviews_bp = Blueprint("reviews", __name__)

_VALID_DECISIONS = frozenset({"approved", "rejected", "request_changes"})


@reviews_bp.route("/<solution_id>")
def review(solution_id: str):
    """Show the review page for a solution."""
    solution_repo = current_app.extensions["solution_repo"]
    issue_repo = current_app.extensions["issue_repo"]
    review_repo = current_app.extensions["review_repo"]

    solution = solution_repo.get(solution_id)
    if solution is None:
        abort(404)

    issue = issue_repo.get(solution.issue_id)
    prior_reviews = review_repo.list_by_solution(solution_id)

    return render_template(
        "reviews/review.html",
        solution=solution,
        issue=issue,
        prior_reviews=prior_reviews,
    )


@reviews_bp.route("/<solution_id>/submit", methods=["POST"])
def submit_review(solution_id: str):
    """Submit a review decision for a solution."""
    solution_repo = current_app.extensions["solution_repo"]
    review_repo = current_app.extensions["review_repo"]
    issue_repo = current_app.extensions["issue_repo"]

    solution = solution_repo.get(solution_id)
    if solution is None:
        abort(404)

    decision = request.form.get("decision", "")
    feedback = request.form.get("feedback", "")
    reviewer = request.form.get("reviewer", "anonymous")

    if decision not in _VALID_DECISIONS:
        abort(400)

    # Create review record
    review_obj = Review(
        id=uuid.uuid4().hex[:12],
        solution_id=solution_id,
        issue_id=solution.issue_id,
        reviewer=reviewer,
        decision=ReviewDecision(decision),
        feedback=feedback,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    review_repo.create(review_obj)

    # Update solution and issue status based on decision
    if decision == "approved":
        solution_repo.update_status(solution_id, SolutionStatus.APPROVED)
        issue_repo.update_status(solution.issue_id, IssueStatus.APPROVED)
    elif decision == "rejected":
        solution_repo.update_status(solution_id, SolutionStatus.REJECTED)
        issue_repo.update_status(solution.issue_id, IssueStatus.REJECTED)
    elif decision == "request_changes":
        solution_repo.update_status(solution_id, SolutionStatus.EDITED)

    # If QueueInterviewer is available, respond to the pipeline
    interviewer = current_app.extensions.get("interviewer")
    if interviewer is not None:
        from attractor.model.question import Answer, Option

        label_map = {
            "approved": "Approve",
            "rejected": "Reject",
            "request_changes": "Request Changes",
        }
        answer = Answer(
            selected_option=Option(key=decision, label=label_map[decision]),
            text=feedback,
        )
        interviewer.respond(answer)

    return redirect(url_for("dashboard.index"))


@reviews_bp.route("/pending")
def pending():
    """Show issues awaiting review."""
    issue_repo = current_app.extensions["issue_repo"]
    solution_repo = current_app.extensions["solution_repo"]

    issues = issue_repo.list_by_status(IssueStatus.AWAITING_REVIEW)
    # Get latest solution for each issue
    review_items: list[dict] = []
    for issue in issues:
        solution = solution_repo.get_latest_for_issue(issue.id)
        if solution:
            review_items.append({"issue": issue, "solution": solution})

    return render_template("reviews/pending.html", review_items=review_items)
