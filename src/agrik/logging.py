"""Logging setup.

Loads a ``dictConfig`` from ``config/logging.yaml`` (falling back to a small
sensible default if the file is missing) and exposes :func:`get_logger` for the
rest of the codebase so every module logs through a consistent namespace.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from .settings import get_settings

_DEFAULT_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {"agrik": {"level": "DEBUG", "handlers": ["console"], "propagate": False}},
    "root": {"level": "INFO", "handlers": ["console"]},
}

_configured = False


def setup_logging(force: bool = False) -> logging.Logger:
    """Initialise logging once, from ``config/logging.yaml`` if present.

    Parameters
    ----------
    force:
        Re-apply configuration even if logging was already set up.
    """
    global _configured
    if _configured and not force:
        return logging.getLogger("agrik")

    settings = get_settings()
    settings.ensure_dirs()

    config_path = Path(settings.logging_config_path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or _DEFAULT_CONFIG
        # Ensure the log-file handler's directory exists before dictConfig runs.
        logs_dir = Path(settings.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
    else:
        config = _DEFAULT_CONFIG

    logging.config.dictConfig(config)
    _configured = True
    logging.getLogger("agrik").debug("Logging initialised (config=%s).", config_path)
    return logging.getLogger("agrik")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``agrik`` namespace."""
    setup_logging()
    return logging.getLogger(f"agrik.{name}")
