from __future__ import annotations

from wolverine.model.review import ReviewDecision
from wolverine.model.solution import FileDiff

from tests.test_wolverine.test_web.conftest import (
    seed_issue,
    seed_review,
    seed_solution,
)

SAMPLE_DIFF = FileDiff(
    file_path="src/app.py",
    original_content="old code",
    modified_content="new code",
    diff_text="--- a/src/app.py\n+++ b/src/app.py\n@@ -1,1 +1,1 @@\n-old code\n+new code",
)


class TestSolutionDetail:
    def test_detail_returns_200(self, client, app):
        """GET /solutions/<id> returns 200 for existing solution."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/solutions/sol-1")
        assert response.status_code == 200

    def test_detail_404_for_nonexistent(self, client):
        """GET /solutions/<nonexistent> returns 404."""
        response = client.get("/solutions/nonexistent")
        assert response.status_code == 404

    def test_detail_shows_summary(self, client, app):
        """Detail page shows the solution summary."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1", summary="Fixed the auth bug")

        response = client.get("/solutions/sol-1")
        assert b"Fixed the auth bug" in response.data

    def test_detail_shows_diff_content(self, client, app):
        """Detail page renders diff text for each file."""
        seed_issue(app, id="iss-1")
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            diffs=(SAMPLE_DIFF,),
        )

        response = client.get("/solutions/sol-1")
        assert b"src/app.py" in response.data
        assert b"-old code" in response.data
        assert b"+new code" in response.data

    def test_detail_shows_issue_context(self, client, app):
        """Detail page shows the parent issue title and description."""
        seed_issue(
            app,
            id="iss-1",
            title="Login bug",
            description="500 errors during login",
        )
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/solutions/sol-1")
        assert b"Login bug" in response.data
        assert b"500 errors during login" in response.data

    def test_detail_shows_test_results(self, client, app):
        """Detail page shows test results when present."""
        seed_issue(app, id="iss-1")
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            test_results="All 10 tests pass",
        )

        response = client.get("/solutions/sol-1")
        assert b"All 10 tests pass" in response.data

    def test_detail_shows_reasoning(self, client, app):
        """Detail page shows the solution reasoning."""
        seed_issue(app, id="iss-1")
        seed_solution(
            app,
            id="sol-1",
            issue_id="iss-1",
            reasoning="The bug was caused by a missing null check",
        )

        response = client.get("/solutions/sol-1")
        assert b"The bug was caused by a missing null check" in response.data

    def test_detail_shows_review_link(self, client, app):
        """Detail page contains a link to the review page."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")

        response = client.get("/solutions/sol-1")
        assert b"/reviews/sol-1" in response.data

    def test_detail_shows_prior_reviews(self, client, app):
        """Detail page shows prior reviews for the solution."""
        seed_issue(app, id="iss-1")
        seed_solution(app, id="sol-1", issue_id="iss-1")
        seed_review(
            app,
            id="rev-1",
            solution_id="sol-1",
            issue_id="iss-1",
            reviewer="alice",
            decision=ReviewDecision.REQUEST_CHANGES,
            feedback="Need more tests",
        )

        response = client.get("/solutions/sol-1")
        assert b"alice" in response.data
        assert b"Need more tests" in response.data
