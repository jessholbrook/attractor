"""Tests for signal processing, preview, and deploy endpoints."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.model.solution import FileDiff, Solution, SolutionStatus
from wolverine.web.app import create_app


def _make_signal(signal_id: str = "sig001") -> RawSignal:
    return RawSignal(
        id=signal_id,
        kind=SignalKind.USER_FEEDBACK,
        source=SignalSource.API,
        title="Button doesn't work",
        body="The submit button on the form is broken.",
        received_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_issue(issue_id: str = "iss001", signal_id: str = "sig001") -> Issue:
    return Issue(
        id=issue_id,
        title="Submit button broken",
        description="The submit button handler is not attached.",
        severity=IssueSeverity.HIGH,
        status=IssueStatus.AWAITING_REVIEW,
        category=IssueCategory.BUG,
        signal_ids=(signal_id,),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_solution(
    solution_id: str = "sol001", issue_id: str = "iss001"
) -> Solution:
    return Solution(
        id=solution_id,
        issue_id=issue_id,
        status=SolutionStatus.GENERATED,
        summary="Fix: Submit button broken",
        reasoning="Fixed the onclick handler.",
        diffs=(
            FileDiff(
                file_path="index.html",
                original_content="<html><body>old</body></html>",
                modified_content="<html><body>fixed</body></html>",
                diff_text="--- a/index.html\n+++ b/index.html\n-old\n+fixed",
            ),
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def app():
    a = create_app()
    a.config["TESTING"] = True
    a.extensions["llm_client"] = MagicMock()
    a.extensions["beacon_html"] = "<html><body>beacon</body></html>"
    a.extensions["github_token"] = ""
    return a


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


class TestProcessSignal:
    def test_returns_404_for_missing_signal(self, client):
        resp = client.post("/api/signals/nonexistent/process")
        assert resp.status_code == 404

    def test_returns_503_without_api_key(self, app):
        app.extensions["llm_client"] = None
        with app.test_client() as c:
            signal = _make_signal()
            app.extensions["signal_repo"].create(signal)
            app.extensions["db"].commit()
            resp = c.post(f"/api/signals/{signal.id}/process")
            assert resp.status_code == 503

    @patch("wolverine.web.routes.process.process_signal")
    def test_creates_issue_and_solution(self, mock_process, app, client):
        signal = _make_signal()
        app.extensions["signal_repo"].create(signal)
        app.extensions["db"].commit()

        issue = _make_issue(signal_id=signal.id)
        solution = _make_solution(issue_id=issue.id)
        mock_process.return_value = (issue, solution)

        resp = client.post(f"/api/signals/{signal.id}/process")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["solution_id"] == solution.id
        assert data["issue_id"] == issue.id

    @patch("wolverine.web.routes.process.process_signal")
    def test_persists_issue_in_db(self, mock_process, app, client):
        signal = _make_signal()
        app.extensions["signal_repo"].create(signal)
        app.extensions["db"].commit()

        issue = _make_issue(signal_id=signal.id)
        solution = _make_solution(issue_id=issue.id)
        mock_process.return_value = (issue, solution)

        client.post(f"/api/signals/{signal.id}/process")

        stored_issue = app.extensions["issue_repo"].get(issue.id)
        assert stored_issue is not None
        assert stored_issue.title == "Submit button broken"

    @patch("wolverine.web.routes.process.process_signal")
    def test_persists_solution_in_db(self, mock_process, app, client):
        signal = _make_signal()
        app.extensions["signal_repo"].create(signal)
        app.extensions["db"].commit()

        issue = _make_issue(signal_id=signal.id)
        solution = _make_solution(issue_id=issue.id)
        mock_process.return_value = (issue, solution)

        client.post(f"/api/signals/{signal.id}/process")

        stored = app.extensions["solution_repo"].get(solution.id)
        assert stored is not None
        assert stored.summary == "Fix: Submit button broken"

    @patch("wolverine.web.routes.process.process_signal")
    def test_returns_500_on_processing_error(self, mock_process, app, client):
        signal = _make_signal()
        app.extensions["signal_repo"].create(signal)
        app.extensions["db"].commit()

        mock_process.side_effect = RuntimeError("LLM failed")

        resp = client.post(f"/api/signals/{signal.id}/process")
        assert resp.status_code == 500
        assert "LLM failed" in resp.get_json()["error"]


def _seed_signal_and_issue(app, signal_id="sig001", issue_id="iss001"):
    """Create a signal and issue in the DB, handling foreign keys."""
    signal = _make_signal(signal_id)
    app.extensions["signal_repo"].create(signal)
    issue = _make_issue(issue_id, signal_id)
    app.extensions["issue_repo"].create(issue)
    app.extensions["db"].commit()
    return signal, issue


class TestPreview:
    def test_returns_html(self, app, client):
        _, issue = _seed_signal_and_issue(app)
        solution = _make_solution(issue_id=issue.id)
        app.extensions["solution_repo"].create(solution)
        app.extensions["db"].commit()

        resp = client.get(f"/solutions/{solution.id}/preview")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/html")
        assert b"fixed" in resp.data

    def test_returns_404_for_missing(self, client):
        resp = client.get("/solutions/nonexistent/preview")
        assert resp.status_code == 404

    def test_returns_404_for_no_diffs(self, app, client):
        _, issue = _seed_signal_and_issue(app)
        solution = Solution(
            id="sol_nodiff",
            issue_id=issue.id,
            status=SolutionStatus.GENERATED,
        )
        app.extensions["solution_repo"].create(solution)
        app.extensions["db"].commit()

        resp = client.get("/solutions/sol_nodiff/preview")
        assert resp.status_code == 404


class TestDeploy:
    def test_returns_503_without_github_token(self, app, client):
        _, issue = _seed_signal_and_issue(app)
        solution = Solution(
            id="sol_dep",
            issue_id=issue.id,
            status=SolutionStatus.APPROVED,
            diffs=_make_solution().diffs,
        )
        app.extensions["solution_repo"].create(solution)
        app.extensions["db"].commit()

        resp = client.post("/api/solutions/sol_dep/deploy")
        assert resp.status_code == 503

    def test_returns_404_for_missing(self, app, client):
        app.extensions["github_token"] = "fake-token"
        resp = client.post("/api/solutions/nonexistent/deploy")
        assert resp.status_code == 404

    def test_requires_approved_status(self, app, client):
        app.extensions["github_token"] = "fake-token"
        _, issue = _seed_signal_and_issue(app)
        solution = _make_solution()  # status=GENERATED, not APPROVED
        app.extensions["solution_repo"].create(solution)
        app.extensions["db"].commit()

        resp = client.post(f"/api/solutions/{solution.id}/deploy")
        assert resp.status_code == 400
        assert "approved" in resp.get_json()["error"].lower()


class TestExtractHtml:
    def test_extracts_from_html_marker(self):
        from wolverine.pipeline.processor import _extract_html

        text = "===CLASSIFICATION===\n{}\n===HTML===\n<!DOCTYPE html><html>test</html>"
        assert _extract_html(text) == "<!DOCTYPE html><html>test</html>"

    def test_extracts_from_code_fence(self):
        from wolverine.pipeline.processor import _extract_html

        text = "```html\n<!DOCTYPE html><html>test</html>\n```"
        assert _extract_html(text) == "<!DOCTYPE html><html>test</html>"

    def test_extracts_raw_doctype(self):
        from wolverine.pipeline.processor import _extract_html

        text = "<!DOCTYPE html><html>test</html>"
        assert _extract_html(text) == "<!DOCTYPE html><html>test</html>"

    def test_strips_whitespace(self):
        from wolverine.pipeline.processor import _extract_html

        text = "  \n<!DOCTYPE html><html>test</html>\n  "
        assert _extract_html(text) == "<!DOCTYPE html><html>test</html>"

    def test_returns_text_as_fallback(self):
        from wolverine.pipeline.processor import _extract_html

        text = "no html here"
        assert _extract_html(text) == "no html here"


class TestParseClassification:
    def test_parses_from_marker(self):
        from wolverine.pipeline.processor import _parse_classification

        text = '===CLASSIFICATION===\n{"severity": "high", "category": "bug", "title": "Test", "description": "A bug"}\n===HTML===\n<html></html>'
        result = _parse_classification(text)
        assert result["severity"] == "high"
        assert result["title"] == "Test"

    def test_handles_json_code_fence(self):
        from wolverine.pipeline.processor import _parse_classification

        text = '===CLASSIFICATION===\n```json\n{"severity": "low", "category": "ux_issue", "title": "UX", "description": "Bad UX"}\n```\n===HTML===\n<html></html>'
        result = _parse_classification(text)
        assert result["severity"] == "low"

    def test_fallback_finds_json_object(self):
        from wolverine.pipeline.processor import _parse_classification

        text = 'Here is the classification: {"severity": "medium", "category": "bug", "title": "Bug", "description": "A bug"}\n\n<!DOCTYPE html>'
        result = _parse_classification(text)
        assert result["severity"] == "medium"

    def test_returns_empty_on_no_json(self):
        from wolverine.pipeline.processor import _parse_classification

        result = _parse_classification("no json here at all")
        assert result == {}
