"""Classification and diagnosis prompts and schemas for the Wolverine pipeline."""
from __future__ import annotations

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "severity": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low"],
            "description": "How severe is this issue?",
        },
        "category": {
            "type": "string",
            "enum": [
                "bug",
                "missing_content",
                "ux_issue",
                "performance",
                "accessibility",
                "other",
            ],
            "description": "What category of issue is this?",
        },
        "title": {
            "type": "string",
            "description": "A concise title for the issue (max 100 chars)",
        },
        "description": {
            "type": "string",
            "description": "A clear description of the issue",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Relevant tags for this issue",
        },
        "is_duplicate": {
            "type": "boolean",
            "description": "Whether this appears to be a duplicate of a known issue",
        },
    },
    "required": ["severity", "category", "title", "description", "tags", "is_duplicate"],
}

CLASSIFY_SYSTEM = """\
You are a software issue classifier. Given a signal (user feedback, error log, etc.), \
classify it by severity, category, and generate a clear title and description.

Severity guide:
- critical: System is down, data loss, security vulnerability
- high: Major feature broken, many users affected
- medium: Feature partially broken, workaround exists
- low: Minor cosmetic issue, enhancement request

Categories:
- bug: Code defect causing incorrect behavior
- missing_content: Help docs, tooltips, or other content missing
- ux_issue: Confusing UI, poor navigation, accessibility problem
- performance: Slow loading, high resource usage
- accessibility: Screen reader issues, keyboard navigation, contrast
- other: Doesn't fit other categories"""

DIAGNOSE_SYSTEM = """\
You are a software diagnostician. Given an issue description, analyze the root cause \
and identify which files in the codebase are likely affected.

Provide:
1. A clear root cause analysis
2. List of affected files (full paths)
3. A recommended fix strategy"""
