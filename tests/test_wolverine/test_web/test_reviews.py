from __future__ import annotations

from wolverine.model.issue import IssueStatus
from wolverine.model.review import ReviewDecision
from wolverine.model.solution import FileDiff, SolutionStatus

from tests.test_wolverine.test_web.conftest import (
    seed_issue,
    seed_review,
    seed_solution,
)

SAMPLE_DIFF = FileDiff(
    file_path="src/auth.py",
    original_content="old code",
    modified_content="new code",
    diff_text="--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,1 +1,1 @@\n-old code\n+new code",
)


class TestReviewPage:
    def test_review_returns_200(self, client, app):
        """GET /reviews/<solution_id> returns 200 for existing solution."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/reviews/sol-1")
        assert response.status_code == 200

    def test_review_404_for_nonexistent(self, client):
        """GET /reviews/<nonexistent> returns 404."""
        response = client.get("/reviews/nonexistent")
        assert response.status_code == 404

    def test_review_shows_issue_context(self, client, app):
        """Review page shows issue title and description."""
        seed_issue(
            app,
            id="iss-1",
            title="Auth failure",
            description="SSO login broken",
        )
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/reviews/sol-1")
        assert b"Auth failure" in response.data
        assert b"SSO login broken" in response.data

    def test_review_shows_diff(self, client, app):
        """Review page renders the diff view."""
        seed_issue(app, id="iss-1")
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            diffs=(SAMPLE_DIFF,),
        )

        response = client.get("/reviews/sol-1")
        assert b"src/auth.py" in response.data
        assert b"-old code" in response.data
        assert b"+new code" in response.data

    def test_review_shows_action_buttons(self, client, app):
        """Review page shows Approve, Request Changes, and Reject buttons."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/reviews/sol-1")
        assert b"Approve" in response.data
        assert b"Request Changes" in response.data
        assert b"Reject" in response.data

    def test_review_shows_solution_summary(self, client, app):
        """Review page shows the solution summary."""
        seed_issue(app, id="iss-1")
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            summary="Added null check to auth handler",
        )

        response = client.get("/reviews/sol-1")
        assert b"Added null check to auth handler" in response.data

    def test_review_shows_prior_reviews(self, client, app):
        """Review page shows prior review history."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")
        seed_review(
            app,
            id="rev-1",
            solution_id="sol-1",
            issue_id="iss-1",
            reviewer="bob",
            decision=ReviewDecision.REQUEST_CHANGES,
            feedback="Add error handling",
        )

        response = client.get("/reviews/sol-1")
        assert b"bob" in response.data
        assert b"Add error handling" in response.data


class TestSubmitReview:
    def test_approve_updates_solution_status(self, client, app):
        """Submitting approve sets solution status to APPROVED."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "tester"},
            follow_redirects=False,
        )
        assert response.status_code == 302

        solution = app.extensions["solution_repo"].get("sol-1")
        assert solution.status == SolutionStatus.APPROVED

    def test_approve_updates_issue_status(self, client, app):
        """Submitting approve sets issue status to APPROVED."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "tester"},
        )

        issue = app.extensions["issue_repo"].get("iss-1")
        assert issue.status == IssueStatus.APPROVED

    def test_reject_updates_statuses(self, client, app):
        """Submitting reject sets solution to REJECTED and issue to REJECTED."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        client.post(
            "/reviews/sol-1/submit",
            data={"decision": "rejected", "reviewer": "tester"},
        )

        solution = app.extensions["solution_repo"].get("sol-1")
        issue = app.extensions["issue_repo"].get("iss-1")
        assert solution.status == SolutionStatus.REJECTED
        assert issue.status == IssueStatus.REJECTED

    def test_request_changes_updates_solution_to_edited(self, client, app):
        """Submitting request_changes sets solution status to EDITED."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        client.post(
            "/reviews/sol-1/submit",
            data={"decision": "request_changes", "reviewer": "tester"},
        )

        solution = app.extensions["solution_repo"].get("sol-1")
        assert solution.status == SolutionStatus.EDITED

    def test_submit_stores_feedback(self, client, app):
        """Submitted feedback is persisted in the review record."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        client.post(
            "/reviews/sol-1/submit",
            data={
                "decision": "request_changes",
                "reviewer": "tester",
                "feedback": "Please add error handling",
            },
        )

        reviews = app.extensions["review_repo"].list_by_solution("sol-1")
        assert len(reviews) == 1
        assert reviews[0].feedback == "Please add error handling"
        assert reviews[0].decision == ReviewDecision.REQUEST_CHANGES

    def test_submit_invalid_decision_returns_400(self, client, app):
        """Submitting an invalid decision returns 400."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.post(
            "/reviews/sol-1/submit",
            data={"decision": "invalid_decision", "reviewer": "tester"},
        )
        assert response.status_code == 400

    def test_submit_nonexistent_solution_returns_404(self, client):
        """Submitting a review for nonexistent solution returns 404."""
        response = client.post(
            "/reviews/nonexistent/submit",
            data={"decision": "approved", "reviewer": "tester"},
        )
        assert response.status_code == 404

    def test_submit_redirects_to_dashboard(self, client, app):
        """Successful review submission redirects to the dashboard."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "tester"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert response.headers["Location"] == "/"

    def test_submit_stores_reviewer_name(self, client, app):
        """Review stores the reviewer name from the form."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "alice"},
        )

        reviews = app.extensions["review_repo"].list_by_solution("sol-1")
        assert len(reviews) == 1
        assert reviews[0].reviewer == "alice"


class TestPendingReviews:
    def test_pending_returns_200(self, client):
        """GET /reviews/pending returns 200."""
        response = client.get("/reviews/pending")
        assert response.status_code == 200

    def test_pending_shows_awaiting_review_issues(self, client, app):
        """Pending page lists issues that are awaiting review."""
        seed_issue(
            app,
            id="iss-1",
            title="Auth bug",
            status=IssueStatus.AWAITING_REVIEW,
        )
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            summary="Fix auth handler",
        )

        response = client.get("/reviews/pending")
        assert b"Auth bug" in response.data
        assert b"Fix auth handler" in response.data

    def test_pending_excludes_non_awaiting_issues(self, client, app):
        """Pending page does not show issues that are not awaiting review."""
        seed_issue(
            app,
            id="iss-1",
            title="Auth bug",
            status=IssueStatus.AWAITING_REVIEW,
        )
        seed_solution(app, id="sol-1", issue_id="iss-1")
        seed_issue(
            app,
            id="iss-2",
            title="Closed bug",
            status=IssueStatus.CLOSED,
        )

        response = client.get("/reviews/pending")
        assert b"Auth bug" in response.data
        assert b"Closed bug" not in response.data

    def test_pending_empty_message(self, client):
        """Pending page shows message when no items await review."""
        response = client.get("/reviews/pending")
        assert b"No solutions awaiting review" in response.data

    def test_pending_has_review_link(self, client, app):
        """Pending page includes a link to the review page for each item."""
        seed_issue(
            app,
            id="iss-1",
            title="Auth bug",
            status=IssueStatus.AWAITING_REVIEW,
        )
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/reviews/pending")
        assert b"/reviews/sol-1" in response.data


class TestQueueInterviewerIntegration:
    def test_submit_with_interviewer_responds(self, db, app):
        """When interviewer is set, submitting a review calls respond()."""
        from attractor.interviewer.queue_interviewer import QueueInterviewer

        interviewer = QueueInterviewer()
        app.extensions["interviewer"] = interviewer

        # Seed data
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        test_client = app.test_client()
        test_client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "tester"},
        )

        # Check that an answer was placed on the queue
        answer = interviewer.answer_queue.get(timeout=1)
        assert answer.selected_option is not None
        assert answer.selected_option.key == "approved"
        assert answer.selected_option.label == "Approve"

    def test_submit_without_interviewer_works(self, client, app):
        """Submit works fine when no interviewer is configured."""
        # Ensure no interviewer extension is present
        assert "interviewer" not in app.extensions

        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.post(
            "/reviews/sol-1/submit",
            data={"decision": "approved", "reviewer": "tester"},
            follow_redirects=False,
        )
        assert response.status_code == 302
