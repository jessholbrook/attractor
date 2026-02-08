from __future__ import annotations

from tests.test_wolverine.test_web.conftest import seed_signal


class TestSignalList:
    def test_list_returns_200(self, client):
        """GET /signals/ returns 200."""
        response = client.get("/signals/")
        assert response.status_code == 200

    def test_list_shows_no_signals_message(self, client):
        """Shows message when no signals exist."""
        response = client.get("/signals/")
        assert b"No signals yet" in response.data

    def test_list_shows_signals_when_populated(self, client, app):
        """Shows signal titles in list."""
        seed_signal(app, id="sig-1", title="Error in payment module")
        seed_signal(app, id="sig-2", title="Crash on startup")

        response = client.get("/signals/")
        assert b"Error in payment module" in response.data
        assert b"Crash on startup" in response.data

    def test_list_contains_heading(self, client):
        """Signal list has the Signals heading."""
        response = client.get("/signals/")
        assert b"Signals" in response.data


class TestSignalSubmit:
    def test_submit_form_get_returns_200(self, client):
        """GET /signals/submit returns the submission form."""
        response = client.get("/signals/submit")
        assert response.status_code == 200

    def test_submit_form_contains_fields(self, client):
        """Submit form contains title, body, and kind fields."""
        response = client.get("/signals/submit")
        assert b'name="title"' in response.data
        assert b'name="body"' in response.data
        assert b'name="kind"' in response.data

    def test_submit_post_creates_signal_and_redirects(self, client, app):
        """POST /signals/submit creates a signal and redirects to list."""
        response = client.post(
            "/signals/submit",
            data={"title": "New error found", "body": "Details here", "kind": "manual"},
            follow_redirects=False,
        )
        assert response.status_code == 302
        assert "/signals/" in response.headers["Location"]

        # Verify signal was created
        signals = app.extensions["signal_repo"].list_all()
        assert len(signals) == 1
        assert signals[0].title == "New error found"
        assert signals[0].body == "Details here"

    def test_submit_post_signal_appears_in_list(self, client):
        """Submitted signal appears in the signal list."""
        client.post(
            "/signals/submit",
            data={"title": "Server crash report", "body": "Segfault at 0x0", "kind": "error_log"},
        )

        response = client.get("/signals/")
        assert b"Server crash report" in response.data

    def test_submit_post_with_different_kind(self, client, app):
        """POST with different signal kind values works."""
        client.post(
            "/signals/submit",
            data={"title": "User complaint", "body": "App is slow", "kind": "user_feedback"},
        )

        signals = app.extensions["signal_repo"].list_all()
        assert len(signals) == 1
        assert signals[0].kind.value == "user_feedback"
        assert signals[0].source.value == "form"

    def test_submit_form_has_kind_options(self, client):
        """Submit form has all SignalKind options."""
        response = client.get("/signals/submit")
        assert b"error_log" in response.data
        assert b"user_feedback" in response.data
        assert b"support_ticket" in response.data
        assert b"manual" in response.data
