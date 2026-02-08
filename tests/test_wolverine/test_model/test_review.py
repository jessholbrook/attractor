from __future__ import annotations

import pytest

from wolverine.model.review import Review, ReviewComment, ReviewDecision


class TestReviewDecision:
    def test_all_values(self) -> None:
        expected = {"approved", "rejected", "request_changes"}
        assert {v.value for v in ReviewDecision} == expected

    def test_is_str(self) -> None:
        assert isinstance(ReviewDecision.APPROVED, str)
        assert ReviewDecision.APPROVED == "approved"

    def test_string_comparison(self) -> None:
        assert "rejected" == ReviewDecision.REJECTED


class TestReviewComment:
    def test_construction(self) -> None:
        comment = ReviewComment(
            file_path="src/app.py",
            line_number=42,
            comment="This variable should be renamed",
        )
        assert comment.file_path == "src/app.py"
        assert comment.line_number == 42
        assert comment.comment == "This variable should be renamed"

    def test_frozen_immutability(self) -> None:
        comment = ReviewComment(file_path="f.py", line_number=1, comment="c")
        with pytest.raises(AttributeError):
            comment.line_number = 2  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(file_path="f.py", line_number=10, comment="fix this")
        assert ReviewComment(**kwargs) == ReviewComment(**kwargs)


class TestReview:
    def test_construction_required_fields(self) -> None:
        review = Review(
            id="rev-001",
            solution_id="sol-001",
            issue_id="iss-001",
            reviewer="human",
            decision=ReviewDecision.APPROVED,
        )
        assert review.id == "rev-001"
        assert review.solution_id == "sol-001"
        assert review.issue_id == "iss-001"
        assert review.reviewer == "human"
        assert review.decision == ReviewDecision.APPROVED

    def test_default_values(self) -> None:
        review = Review(
            id="rev-002",
            solution_id="sol-002",
            issue_id="iss-002",
            reviewer="auto",
            decision=ReviewDecision.REJECTED,
        )
        assert review.feedback == ""
        assert review.comments == ()
        assert review.created_at == ""

    def test_construction_with_comments(self) -> None:
        comments = (
            ReviewComment(file_path="a.py", line_number=1, comment="fix"),
            ReviewComment(file_path="b.py", line_number=5, comment="rename"),
        )
        review = Review(
            id="rev-003",
            solution_id="sol-003",
            issue_id="iss-003",
            reviewer="senior-dev",
            decision=ReviewDecision.REQUEST_CHANGES,
            feedback="Needs work",
            comments=comments,
        )
        assert len(review.comments) == 2
        assert review.comments[0].file_path == "a.py"
        assert review.feedback == "Needs work"

    def test_frozen_immutability(self) -> None:
        review = Review(
            id="rev-004",
            solution_id="sol-004",
            issue_id="iss-004",
            reviewer="bot",
            decision=ReviewDecision.APPROVED,
        )
        with pytest.raises(AttributeError):
            review.decision = ReviewDecision.REJECTED  # type: ignore[misc]
        with pytest.raises(AttributeError):
            review.feedback = "changed"  # type: ignore[misc]

    def test_hashable(self) -> None:
        review = Review(
            id="rev-005",
            solution_id="sol-005",
            issue_id="iss-005",
            reviewer="dev",
            decision=ReviewDecision.APPROVED,
        )
        assert isinstance(hash(review), int)
        s = {review}
        assert review in s
