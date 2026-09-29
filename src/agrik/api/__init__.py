"""Serving layer (FastAPI) - a thin, read-only API over pipeline artefacts.

Optional dependency extra: ``pip install -e ".[api]"`` (fastapi + uvicorn).
Run with ``agrik-serve`` or ``python -m agrik.api``. The API never imports
ingestion or modelling logic - it only reads what ``python -m agrik`` wrote.
"""

from .app import create_app

__all__ = ["create_app"]
