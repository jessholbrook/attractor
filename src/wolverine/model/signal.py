from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SignalKind(StrEnum):
    ERROR_LOG = "error_log"
    USER_FEEDBACK = "user_feedback"
    SUPPORT_TICKET = "support_ticket"
    UX_RESEARCH = "ux_research"
    SOCIAL_MEDIA = "social_media"
    MANUAL = "manual"


class SignalSource(StrEnum):
    SENTRY = "sentry"
    DATADOG = "datadog"
    FORM = "form"
    CSV = "csv"
    API = "api"
    CLI = "cli"


@dataclass(frozen=True)
class RawSignal:
    id: str
    kind: SignalKind
    source: SignalSource
    title: str
    body: str
    received_at: str  # ISO 8601
    metadata: dict[str, str] = field(default_factory=dict, hash=False, compare=False)
    raw_payload: str = ""
