"""Console entry point: ``agrik-serve`` / ``python -m agrik.api``.

Host and port come from settings (``AGRIK_API_HOST`` / ``AGRIK_API_PORT``,
defaults 127.0.0.1:8000). Uvicorn is imported lazily so the core package
never requires the optional ``[api]`` extra.
"""

from __future__ import annotations

from ..logging import setup_logging
from ..settings import get_settings


def main() -> int:
    import uvicorn

    from .app import create_app

    setup_logging()
    s = get_settings()
    uvicorn.run(create_app(), host=s.api_host, port=s.api_port,
                log_level=s.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
