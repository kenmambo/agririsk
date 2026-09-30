"""Console entry point: ``agrik-serve`` / ``python -m agrik.api``.

Host and port come from settings (``AGRIK_API_HOST`` / ``AGRIK_API_PORT``,
defaults 127.0.0.1:8000). On PaaS hosts (Render/Fly/heroku-style) the platform
injects a ``PORT`` environment variable: if ``AGRIK_API_PORT`` is not set
explicitly, ``PORT`` is honored so the same image deploys unchanged. Uvicorn is
imported lazily so the core package never requires the optional ``[api]`` extra.
"""

from __future__ import annotations

import os

from ..logging import setup_logging
from ..settings import Settings, get_settings


def resolve_port(s: Settings, environ: dict[str, str] | None = None) -> int:
    """Explicit AGRIK_API_PORT wins; then the PaaS-injected PORT; else default."""
    env = os.environ if environ is None else environ
    if "AGRIK_API_PORT" in env:
        return s.api_port
    pas = env.get("PORT", "")
    if pas.isdigit() and 0 < int(pas) < 65536:
        return int(pas)
    return s.api_port


def main() -> int:
    import uvicorn

    from .app import create_app

    setup_logging()
    s = get_settings()
    uvicorn.run(create_app(), host=s.api_host, port=resolve_port(s),
                log_level=s.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
