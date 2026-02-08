"""Diagnostic model: structured validation messages for graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Severity level for a diagnostic message."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class Diagnostic:
    """A single validation finding about the pipeline graph.

    Attributes:
        rule: Identifier for the validation rule that produced this diagnostic.
        severity: How serious the issue is.
        message: Human-readable description of the problem.
        node_id: The node involved, if applicable.
        edge: The (from, to) node pair involved, if applicable.
        fix: Suggested remediation, if available.
    """

    rule: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge: tuple[str, str] | None = None
    fix: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    @property
    def is_warning(self) -> bool:
        return self.severity is Severity.WARNING

    def __str__(self) -> str:
        location = ""
        if self.node_id:
            location = f" [node={self.node_id}]"
        elif self.edge:
            location = f" [edge={self.edge[0]}->{self.edge[1]}]"
        return f"{self.severity.value}{location}: {self.message}"
