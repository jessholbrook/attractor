from __future__ import annotations

import pytest

from wolverine.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestAboutPage:
    def test_returns_200(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_contains_heading(self, client):
        resp = client.get("/about")
        assert b"About Wolverine" in resp.data

    def test_contains_architecture_section(self, client):
        resp = client.get("/about")
        assert b"Architecture" in resp.data

    def test_contains_package_names(self, client):
        resp = client.get("/about")
        for pkg in [b"unified_llm", b"agent_loop", b"attractor"]:
            assert pkg in resp.data

    def test_contains_stats_section(self, client):
        resp = client.get("/about")
        assert b"Project Stats" in resp.data

    def test_contains_how_it_works(self, client):
        resp = client.get("/about")
        assert b"How It Works" in resp.data

    def test_nav_has_about_link(self, client):
        resp = client.get("/about")
        assert b'href="/about"' in resp.data
