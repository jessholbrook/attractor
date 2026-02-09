from __future__ import annotations

from flask import Blueprint, render_template

about_bp = Blueprint("about", __name__)


@about_bp.route("/about")
def about():
    """About page describing the Wolverine system."""
    return render_template("about.html")
