from __future__ import annotations

from flask import Flask

from wolverine.store.db import Database
from wolverine.store.migrations import run_migrations
from wolverine.store.repositories import (
    IssueRepository,
    ReviewRepository,
    RunRepository,
    SignalRepository,
    SolutionRepository,
)


def create_app(
    db: Database | None = None,
    config: dict | None = None,
    interviewer: object | None = None,
) -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.config.update(config or {})

    # Store db and repos on app for access in routes
    if db is None:
        db = Database(":memory:")
        db.connect()
        run_migrations(db)

    app.extensions["db"] = db
    app.extensions["signal_repo"] = SignalRepository(db)
    app.extensions["issue_repo"] = IssueRepository(db)
    app.extensions["solution_repo"] = SolutionRepository(db)
    app.extensions["review_repo"] = ReviewRepository(db)
    app.extensions["run_repo"] = RunRepository(db)

    if interviewer is not None:
        app.extensions["interviewer"] = interviewer

    # Register blueprints
    from wolverine.web.routes.dashboard import dashboard_bp
    from wolverine.web.routes.issues import issues_bp
    from wolverine.web.routes.reviews import reviews_bp
    from wolverine.web.routes.signals import signals_bp
    from wolverine.web.routes.solutions import solutions_bp
    from wolverine.web.routes.api import api_bp
    from wolverine.web.routes.about import about_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(issues_bp, url_prefix="/issues")
    app.register_blueprint(solutions_bp, url_prefix="/solutions")
    app.register_blueprint(reviews_bp, url_prefix="/reviews")
    app.register_blueprint(signals_bp, url_prefix="/signals")
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(about_bp)

    return app
