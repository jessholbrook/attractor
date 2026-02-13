"""Vercel serverless entry point for Wolverine."""
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.web.app import create_app

# In-memory DB for serverless demo
db = Database(":memory:")
db.connect()
run_migrations(db)

# Seed a few sample signals so the demo isn't empty
import uuid
from datetime import datetime, timezone
from wolverine.model.signal import RawSignal, SignalKind, SignalSource
from wolverine.store.repositories import SignalRepository

repo = SignalRepository(db)
samples = [
    RawSignal(
        id=uuid.uuid4().hex[:12],
        kind=SignalKind.USER_FEEDBACK,
        source=SignalSource.FORM,
        title="Login page crashes on Safari",
        body="Multiple users report the login form freezes after clicking submit on Safari 17. Console shows a TypeError in the auth handler.",
        received_at=datetime.now(timezone.utc).isoformat(),
    ),
    RawSignal(
        id=uuid.uuid4().hex[:12],
        kind=SignalKind.ERROR_LOG,
        source=SignalSource.CLI,
        title="NullPointerException in PaymentService",
        body="java.lang.NullPointerException at PaymentService.processRefund(PaymentService.java:142). Occurs when refund amount exceeds original charge.",
        received_at=datetime.now(timezone.utc).isoformat(),
    ),
    RawSignal(
        id=uuid.uuid4().hex[:12],
        kind=SignalKind.UX_RESEARCH,
        source=SignalSource.FORM,
        title="Users can't find the export button",
        body="In usability testing, 4 out of 6 participants failed to locate the data export feature. They expected it in the toolbar but it's buried in Settings > Advanced.",
        received_at=datetime.now(timezone.utc).isoformat(),
    ),
]
for signal in samples:
    repo.create(signal)
db.commit()

app = create_app(db=db)
