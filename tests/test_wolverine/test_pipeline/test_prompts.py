from __future__ import annotations

from wolverine.model.issue import IssueCategory, IssueSeverity
from wolverine.pipeline.prompts import CLASSIFY_SCHEMA, CLASSIFY_SYSTEM, DIAGNOSE_SYSTEM


class TestClassifySchema:
    def test_has_required_fields(self) -> None:
        required = CLASSIFY_SCHEMA["required"]
        expected = ["severity", "category", "title", "description", "tags", "is_duplicate"]
        assert set(required) == set(expected)

    def test_severity_enum_matches_issue_severity(self) -> None:
        schema_values = CLASSIFY_SCHEMA["properties"]["severity"]["enum"]
        model_values = [s.value for s in IssueSeverity]
        assert set(schema_values) == set(model_values)

    def test_category_enum_matches_issue_category(self) -> None:
        schema_values = CLASSIFY_SCHEMA["properties"]["category"]["enum"]
        model_values = [c.value for c in IssueCategory]
        assert set(schema_values) == set(model_values)

    def test_tags_is_array_of_strings(self) -> None:
        tags_prop = CLASSIFY_SCHEMA["properties"]["tags"]
        assert tags_prop["type"] == "array"
        assert tags_prop["items"]["type"] == "string"

    def test_is_duplicate_is_boolean(self) -> None:
        dup_prop = CLASSIFY_SCHEMA["properties"]["is_duplicate"]
        assert dup_prop["type"] == "boolean"

    def test_title_is_string(self) -> None:
        title_prop = CLASSIFY_SCHEMA["properties"]["title"]
        assert title_prop["type"] == "string"

    def test_description_is_string(self) -> None:
        desc_prop = CLASSIFY_SCHEMA["properties"]["description"]
        assert desc_prop["type"] == "string"


class TestClassifySystem:
    def test_is_non_empty_string(self) -> None:
        assert isinstance(CLASSIFY_SYSTEM, str)
        assert len(CLASSIFY_SYSTEM) > 0

    def test_mentions_severity_levels(self) -> None:
        for level in ["critical", "high", "medium", "low"]:
            assert level in CLASSIFY_SYSTEM

    def test_mentions_categories(self) -> None:
        for cat in ["bug", "missing_content", "ux_issue", "performance", "accessibility", "other"]:
            assert cat in CLASSIFY_SYSTEM


class TestDiagnoseSystem:
    def test_is_non_empty_string(self) -> None:
        assert isinstance(DIAGNOSE_SYSTEM, str)
        assert len(DIAGNOSE_SYSTEM) > 0

    def test_mentions_root_cause(self) -> None:
        assert "root cause" in DIAGNOSE_SYSTEM

    def test_mentions_affected_files(self) -> None:
        assert "affected files" in DIAGNOSE_SYSTEM.lower()
