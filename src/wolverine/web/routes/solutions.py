from __future__ import annotations

from flask import Blueprint, abort, current_app, render_template

solutions_bp = Blueprint("solutions", __name__)


@solutions_bp.route("/<solution_id>")
def detail(solution_id: str):
    """Show solution detail with diffs, reasoning, and issue context."""
    solution_repo = current_app.extensions["solution_repo"]
    issue_repo = current_app.extensions["issue_repo"]
    review_repo = current_app.extensions["review_repo"]

    solution = solution_repo.get(solution_id)
    if solution is None:
        abort(404)

    issue = issue_repo.get(solution.issue_id)
    reviews = review_repo.list_by_solution(solution_id)

    return render_template(
        "solutions/detail.html",
        solution=solution,
        issue=issue,
        reviews=reviews,
    )
