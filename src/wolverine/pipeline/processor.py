"""Signal-to-solution processor for the Beacon demo."""
from __future__ import annotations

import difflib
import json
import re
import uuid
from datetime import datetime, timezone

from unified_llm import Client as UnifiedClient
from unified_llm import generate

from wolverine.model.issue import Issue, IssueCategory, IssueSeverity, IssueStatus
from wolverine.model.solution import FileDiff, Solution, SolutionStatus
from wolverine.pipeline.prompts import CLASSIFY_SYSTEM, HEAL_BEACON_SYSTEM


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _extract_html(text: str) -> str:
    """Extract HTML from Claude's response, handling ```html markers."""
    match = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"(<!DOCTYPE html>.*</html>)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def process_signal(
    signal,
    beacon_html: str,
    client: UnifiedClient,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[Issue, Solution]:
    """Classify a signal, create an issue, and generate a fixed Beacon HTML."""
    now = datetime.now(timezone.utc).isoformat()

    # Step 1: Classify (using generate + manual JSON parse for Anthropic compat)
    classify_result = generate(
        model,
        prompt=f"""Classify this signal and respond with ONLY a JSON object (no markdown, no explanation):

Title: {signal.title}
Body: {signal.body}

Return JSON with these exact keys:
- "severity": one of "critical", "high", "medium", "low"
- "category": one of "bug", "missing_content", "ux_issue", "performance", "accessibility", "other"
- "title": concise issue title (max 100 chars)
- "description": clear description of the issue
- "tags": array of relevant tag strings
- "is_duplicate": boolean""",
        system=CLASSIFY_SYSTEM,
        client=client,
    )
    # Extract JSON from response (handle possible markdown wrapping)
    raw = classify_result.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    classification = json.loads(raw)

    # Step 2: Create Issue
    issue = Issue(
        id=_generate_id(),
        title=classification["title"],
        description=classification["description"],
        severity=IssueSeverity(classification["severity"]),
        status=IssueStatus.HEALING,
        category=IssueCategory(classification["category"]),
        signal_ids=(signal.id,),
        tags=tuple(classification.get("tags", [])),
        created_at=now,
        updated_at=now,
    )

    # Step 3: Generate fix
    heal_result = generate(
        model,
        prompt=f"""Signal: {signal.title}
{signal.body}

Diagnosis: {issue.description}
Severity: {issue.severity}
Category: {issue.category}

Here is the complete current HTML source:

```html
{beacon_html}
```

Generate the complete fixed HTML:""",
        system=HEAL_BEACON_SYSTEM,
        client=client,
        max_tokens=16384,
    )

    fixed_html = _extract_html(heal_result.text)

    # Step 4: Create diff
    diff_lines = list(difflib.unified_diff(
        beacon_html.splitlines(keepends=True),
        fixed_html.splitlines(keepends=True),
        fromfile="a/index.html",
        tofile="b/index.html",
    ))
    diff_text = "".join(diff_lines)

    file_diff = FileDiff(
        file_path="index.html",
        original_content=beacon_html,
        modified_content=fixed_html,
        diff_text=diff_text,
    )

    # Step 5: Create Solution
    solution = Solution(
        id=_generate_id(),
        issue_id=issue.id,
        status=SolutionStatus.GENERATED,
        summary=f"Fix: {issue.title}",
        reasoning=f"Generated fix for: {issue.description}",
        diffs=(file_diff,),
        created_at=now,
        llm_model=model,
    )

    return issue, solution
