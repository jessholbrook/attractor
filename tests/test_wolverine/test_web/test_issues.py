from __future__ import annotations

from wolverine.model.issue import IssueStatus

from tests.test_wolverine.test_web.conftest import (
    seed_issue,
    seed_signal,
    seed_solution,
)


class TestIssueList:
    def test_list_returns_200_empty(self, client):
        """GET /issues/ returns 200 with no issues."""
        response = client.get("/issues/")
        assert response.status_code == 200

    def test_list_contains_heading(self, client):
        """Issues page has the Issues heading."""
        response = client.get("/issues/")
        assert b"Issues" in response.data

    def test_list_shows_issues_when_populated(self, client, app):
        """Issues list shows issue titles."""
        seed_issue(app, id="iss-1", title="Login bug")
        seed_issue(app, id="iss-2", title="Dashboard crash")

        response = client.get("/issues/")
        assert b"Login bug" in response.data
        assert b"Dashboard crash" in response.data

    def test_list_shows_no_issues_message_when_empty(self, client):
        """Shows message when no issues exist."""
        response = client.get("/issues/")
        assert b"No issues found" in response.data

    def test_status_filter_works(self, client, app):
        """Status filter shows only matching issues."""
        seed_issue(app, id="iss-1", title="New issue", status=IssueStatus.NEW)
        seed_issue(app, id="iss-2", title="Triaged issue", status=IssueStatus.TRIAGED)

        response = client.get("/issues/?status=new")
        assert response.status_code == 200
        assert b"New issue" in response.data
        assert b"Triaged issue" not in response.data

    def test_status_filter_triaged(self, client, app):
        """Status filter for triaged works."""
        seed_issue(app, id="iss-1", title="New issue", status=IssueStatus.NEW)
        seed_issue(app, id="iss-2", title="Triaged issue", status=IssueStatus.TRIAGED)

        response = client.get("/issues/?status=triaged")
        assert response.status_code == 200
        assert b"Triaged issue" in response.data
        assert b"New issue" not in response.data


class TestIssueDetail:
    def test_detail_returns_200(self, client, app):
        """GET /issues/<id> returns 200 for existing issue."""
        seed_issue(app, id="iss-1", title="Login bug")

        response = client.get("/issues/iss-1")
        assert response.status_code == 200

    def test_detail_shows_issue_title(self, client, app):
        """Detail page shows the issue title."""
        seed_issue(app, id="iss-1", title="Login bug")

        response = client.get("/issues/iss-1")
        assert b"Login bug" in response.data

    def test_detail_404_for_nonexistent(self, client):
        """GET /issues/<nonexistent> returns 404."""
        response = client.get("/issues/nonexistent")
        assert response.status_code == 404

    def test_detail_shows_solutions(self, client, app):
        """Detail page shows solution history."""
        seed_issue(app, id="iss-1", title="Login bug")
        seed_solution(app, id="sol-1", issue_id="iss-1", summary="Fix the auth check")

        response = client.get("/issues/iss-1")
        assert b"Fix the auth check" in response.data

    def test_detail_shows_related_signals(self, client, app):
        """Detail page shows related signals."""
        seed_signal(app, id="sig-1", title="Error in auth module")
        seed_issue(app, id="iss-1", title="Login bug", signal_ids=("sig-1",))

        response = client.get("/issues/iss-1")
        assert b"Error in auth module" in response.data

    def test_detail_shows_description(self, client, app):
        """Detail page shows the issue description."""
        seed_issue(
            app,
            id="iss-1",
            title="Login bug",
            description="The login form returns 500 for SSO users.",
        )

        response = client.get("/issues/iss-1")
        assert b"The login form returns 500 for SSO users." in response.data
