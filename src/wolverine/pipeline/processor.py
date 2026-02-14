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


def _parse_classification(text: str) -> dict:
    """Extract JSON classification from the ===CLASSIFICATION=== block."""
    match = re.search(r"===CLASSIFICATION===\s*\n?(.*?)(?:===HTML===|```|$)", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
        # Strip markdown code fence if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # Fallback: try to find any JSON object in the text before the HTML
    match = re.search(r"\{[^{}]*\"severity\"[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _extract_html(text: str) -> str:
    """Extract HTML from Claude's response, handling various markers."""
    # Check for ===HTML=== marker first
    match = re.search(r"===HTML===\s*\n?(.*)", text, re.DOTALL)
    if match:
        html = match.group(1).strip()
        # Remove any trailing ``` if wrapped
        html = re.sub(r"\n?```\s*$", "", html)
        return html
    # Then check ```html fences
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
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[Issue, Solution]:
    """Classify a signal and generate a fixed Beacon HTML in one LLM call."""
    now = datetime.now(timezone.utc).isoformat()

    # Single LLM call: classify + fix in one shot to stay within Vercel timeout
    heal_result = generate(
        model,
        prompt=f"""Bug report:
Title: {signal.title}
Body: {signal.body}

Here is the complete current HTML source:

```html
{beacon_html}
```

First, output a JSON classification block, then the fixed HTML.

OUTPUT FORMAT (follow exactly):
===CLASSIFICATION===
{{"severity": "high", "category": "bug", "title": "...", "description": "..."}}
===HTML===
<!DOCTYPE html>
... complete fixed HTML ...
</html>""",
        system=CLASSIFY_SYSTEM + "\n\n" + HEAL_BEACON_SYSTEM,
        client=client,
        max_tokens=16384,
    )

    # Parse classification JSON and fixed HTML from single response
    text = heal_result.text
    classification = _parse_classification(text)

    # Create Issue
    issue = Issue(
        id=_generate_id(),
        title=classification.get("title", signal.title),
        description=classification.get("description", signal.body),
        severity=IssueSeverity(classification.get("severity", "medium")),
        status=IssueStatus.HEALING,
        category=IssueCategory(classification.get("category", "bug")),
        signal_ids=(signal.id,),
        tags=tuple(classification.get("tags", [])),
        created_at=now,
        updated_at=now,
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
