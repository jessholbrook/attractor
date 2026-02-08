from __future__ import annotations

import json

from wolverine.model.issue import IssueStatus
from wolverine.model.run import RunStatus

from tests.test_wolverine.test_web.conftest import (
    seed_issue,
    seed_run,
    seed_signal,
)


class TestCreateSignalAPI:
    def test_post_creates_signal(self, client, app):
        """POST /api/signals creates a signal and returns 202."""
        response = client.post(
            "/api/signals",
            json={"title": "API error", "body": "500 on /users endpoint"},
        )
        assert response.status_code == 202
        data = response.get_json()
        assert "id" in data
        assert data["status"] == "accepted"

        # Verify signal was persisted
        signals = app.extensions["signal_repo"].list_all()
        assert len(signals) == 1
        assert signals[0].title == "API error"
        assert signals[0].source.value == "api"

    def test_post_missing_title_returns_400(self, client):
        """POST /api/signals without title returns 400."""
        response = client.post(
            "/api/signals",
            json={"body": "Some body text"},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_post_missing_body_returns_400(self, client):
        """POST /api/signals without body returns 400."""
        response = client.post(
            "/api/signals",
            json={"title": "Some title"},
        )
        assert response.status_code == 400

    def test_post_empty_json_returns_400(self, client):
        """POST /api/signals with empty JSON returns 400."""
        response = client.post(
            "/api/signals",
            data="{}",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_post_no_json_returns_error(self, client):
        """POST /api/signals with no JSON body returns an error status."""
        response = client.post("/api/signals", content_type="application/json")
        assert response.status_code == 400

    def test_post_with_kind_and_metadata(self, client, app):
        """POST /api/signals with kind and metadata creates signal correctly."""
        response = client.post(
            "/api/signals",
            json={
                "title": "Sentry error",
                "body": "NullPointerException",
                "kind": "error_log",
                "metadata": {"env": "production", "service": "auth"},
            },
        )
        assert response.status_code == 202

        signals = app.extensions["signal_repo"].list_all()
        assert len(signals) == 1
        assert signals[0].kind.value == "error_log"
        assert signals[0].metadata == {"env": "production", "service": "auth"}


class TestIssueStatusAPI:
    def test_issue_status_returns_status(self, client, app):
        """GET /api/issues/<id>/status returns the issue status."""
        seed_issue(app, id="iss-1", status=IssueStatus.TRIAGED)

        response = client.get("/api/issues/iss-1/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == "iss-1"
        assert data["status"] == "triaged"

    def test_issue_status_404_for_nonexistent(self, client):
        """GET /api/issues/<nonexistent>/status returns 404."""
        response = client.get("/api/issues/nonexistent/status")
        assert response.status_code == 404
        data = response.get_json()
        assert data["error"] == "not found"


class TestRunStatusAPI:
    def test_run_status_returns_status(self, client, app):
        """GET /api/runs/<id>/status returns the run status."""
        seed_signal(app, id="sig-1")
        seed_run(app, id="run-1", signal_id="sig-1", status=RunStatus.INGESTING)

        response = client.get("/api/runs/run-1/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["id"] == "run-1"
        assert data["status"] == "ingesting"

    def test_run_status_404_for_nonexistent(self, client):
        """GET /api/runs/<nonexistent>/status returns 404."""
        response = client.get("/api/runs/nonexistent/status")
        assert response.status_code == 404


class TestDashboardStatsAPI:
    def test_dashboard_stats_empty(self, client):
        """GET /api/dashboard/stats returns empty counts when no data."""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert data["issues"] == {}
        assert data["runs"] == {}

    def test_dashboard_stats_with_data(self, client, app):
        """GET /api/dashboard/stats returns correct counts."""
        seed_issue(app, id="iss-1", status=IssueStatus.NEW)
        seed_issue(app, id="iss-2", status=IssueStatus.NEW)
        seed_issue(app, id="iss-3", status=IssueStatus.TRIAGED)

        seed_signal(app, id="sig-1")
        seed_signal(app, id="sig-2")
        seed_run(app, id="run-1", signal_id="sig-1", status=RunStatus.PENDING)
        seed_run(app, id="run-2", signal_id="sig-2", status=RunStatus.COMPLETED)

        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert data["issues"]["new"] == 2
        assert data["issues"]["triaged"] == 1
        assert data["runs"]["pending"] == 1
        assert data["runs"]["completed"] == 1
