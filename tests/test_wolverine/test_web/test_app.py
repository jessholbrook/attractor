from __future__ import annotations

from flask import Flask

from wolverine.store.db import Database
from wolverine.store.repositories import (
    IssueRepository,
    ReviewRepository,
    RunRepository,
    SignalRepository,
    SolutionRepository,
)
from wolverine.web.app import create_app


class TestCreateApp:
    def test_app_creates_successfully(self, app):
        """App factory returns a Flask application."""
        assert isinstance(app, Flask)

    def test_app_creates_with_default_db(self):
        """App factory creates an in-memory DB when none is provided."""
        app = create_app()
        assert isinstance(app, Flask)
        assert "db" in app.extensions

    def test_db_accessible_via_extensions(self, app, db):
        """Database is stored in app.extensions."""
        assert app.extensions["db"] is db

    def test_repos_accessible_via_extensions(self, app):
        """All repositories are stored in app.extensions."""
        assert isinstance(app.extensions["signal_repo"], SignalRepository)
        assert isinstance(app.extensions["issue_repo"], IssueRepository)
        assert isinstance(app.extensions["solution_repo"], SolutionRepository)
        assert isinstance(app.extensions["review_repo"], ReviewRepository)
        assert isinstance(app.extensions["run_repo"], RunRepository)

    def test_dashboard_blueprint_registered(self, app):
        """Dashboard blueprint is registered at /."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/" in rules

    def test_issues_blueprint_registered(self, app):
        """Issues blueprint is registered at /issues/."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/issues/" in rules

    def test_signals_blueprint_registered(self, app):
        """Signals blueprint is registered at /signals/."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/signals/" in rules

    def test_api_blueprint_registered(self, app):
        """API blueprint is registered at /api/."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/api/signals" in rules
        assert "/api/dashboard/stats" in rules


class TestDashboardRoute:
    def test_dashboard_returns_200(self, client):
        """GET / returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_dashboard_contains_title(self, client):
        """Dashboard page contains the expected heading."""
        response = client.get("/")
        assert b"Dashboard" in response.data

    def test_dashboard_contains_nav(self, client):
        """Dashboard page contains navigation links."""
        response = client.get("/")
        assert b"Wolverine" in response.data
        assert b"Issues" in response.data
        assert b"Signals" in response.data
