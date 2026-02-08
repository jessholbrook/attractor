from __future__ import annotations

import uuid
from datetime import datetime, timezone

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from wolverine.model.signal import RawSignal, SignalKind, SignalSource

signals_bp = Blueprint("signals", __name__)


def _generate_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:12]


@signals_bp.route("/")
def list_signals():
    """List all signals."""
    signal_repo = current_app.extensions["signal_repo"]
    signals = signal_repo.list_all()
    return render_template("signals/list.html", signals=signals)


@signals_bp.route("/submit", methods=["GET", "POST"])
def submit():
    """Show signal submission form (GET) or create a signal (POST)."""
    if request.method == "POST":
        signal = RawSignal(
            id=_generate_id(),
            kind=SignalKind(request.form.get("kind", "manual")),
            source=SignalSource.FORM,
            title=request.form["title"],
            body=request.form["body"],
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        signal_repo = current_app.extensions["signal_repo"]
        signal_repo.create(signal)
        return redirect(url_for("signals.list_signals"))

    return render_template("signals/submit.html")
