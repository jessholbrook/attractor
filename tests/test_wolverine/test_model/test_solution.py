from __future__ import annotations

import pytest

from wolverine.model.solution import FileDiff, Solution, SolutionStatus


class TestSolutionStatus:
    def test_all_values(self) -> None:
        expected = {"generating", "generated", "validated", "failed", "approved", "rejected", "edited"}
        assert {v.value for v in SolutionStatus} == expected

    def test_is_str(self) -> None:
        assert isinstance(SolutionStatus.GENERATING, str)
        assert SolutionStatus.APPROVED == "approved"


class TestFileDiff:
    def test_construction(self) -> None:
        diff = FileDiff(
            file_path="src/app.py",
            original_content="print('hello')",
            modified_content="print('world')",
            diff_text="--- a/src/app.py\n+++ b/src/app.py\n-print('hello')\n+print('world')",
        )
        assert diff.file_path == "src/app.py"
        assert diff.original_content == "print('hello')"
        assert diff.modified_content == "print('world')"
        assert "---" in diff.diff_text

    def test_frozen_immutability(self) -> None:
        diff = FileDiff(
            file_path="f.py",
            original_content="a",
            modified_content="b",
            diff_text="d",
        )
        with pytest.raises(AttributeError):
            diff.file_path = "other.py"  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(file_path="f.py", original_content="a", modified_content="b", diff_text="d")
        assert FileDiff(**kwargs) == FileDiff(**kwargs)

    def test_hashable(self) -> None:
        diff = FileDiff(file_path="f.py", original_content="a", modified_content="b", diff_text="d")
        assert isinstance(hash(diff), int)


class TestSolution:
    def test_construction_required_fields(self) -> None:
        sol = Solution(
            id="sol-001",
            issue_id="iss-001",
            status=SolutionStatus.GENERATING,
        )
        assert sol.id == "sol-001"
        assert sol.issue_id == "iss-001"
        assert sol.status == SolutionStatus.GENERATING

    def test_default_values(self) -> None:
        sol = Solution(id="sol-002", issue_id="iss-002", status=SolutionStatus.GENERATED)
        assert sol.summary == ""
        assert sol.reasoning == ""
        assert sol.diffs == ()
        assert sol.test_results == ""
        assert sol.agent_session_id == ""
        assert sol.created_at == ""
        assert sol.attempt_number == 1
        assert sol.llm_model == ""

    def test_default_token_usage_is_empty_dict(self) -> None:
        sol = Solution(id="sol-003", issue_id="iss-003", status=SolutionStatus.VALIDATED)
        assert sol.token_usage == {}

    def test_token_usage_distinct_per_instance(self) -> None:
        a = Solution(id="a", issue_id="i", status=SolutionStatus.GENERATED)
        b = Solution(id="b", issue_id="i", status=SolutionStatus.GENERATED)
        assert a.token_usage is not b.token_usage

    def test_construction_with_diffs(self) -> None:
        diff = FileDiff(file_path="f.py", original_content="a", modified_content="b", diff_text="d")
        sol = Solution(
            id="sol-004",
            issue_id="iss-004",
            status=SolutionStatus.VALIDATED,
            diffs=(diff,),
        )
        assert len(sol.diffs) == 1
        assert sol.diffs[0].file_path == "f.py"

    def test_frozen_immutability(self) -> None:
        sol = Solution(id="sol-005", issue_id="iss-005", status=SolutionStatus.GENERATING)
        with pytest.raises(AttributeError):
            sol.status = SolutionStatus.FAILED  # type: ignore[misc]
        with pytest.raises(AttributeError):
            sol.summary = "changed"  # type: ignore[misc]

    def test_equality_excludes_token_usage(self) -> None:
        kwargs = dict(id="sol-eq", issue_id="iss-eq", status=SolutionStatus.GENERATED)
        a = Solution(**kwargs, token_usage={"input": 100})
        b = Solution(**kwargs, token_usage={"input": 200})
        assert a == b

    def test_hash_excludes_token_usage(self) -> None:
        kwargs = dict(id="sol-hash", issue_id="iss-hash", status=SolutionStatus.GENERATED)
        a = Solution(**kwargs, token_usage={"input": 100})
        b = Solution(**kwargs, token_usage={"input": 200})
        assert hash(a) == hash(b)
