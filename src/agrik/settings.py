"""Application settings and configuration management.

Uses ``pydantic-settings`` so every value can be overridden by environment
variables (prefix ``AGRIK_``) or a local ``.env`` file, which is the standard
approach for 12-factor style configuration. Paths default to locations relative
to the working directory (the repository root).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the AgriRisk pipeline and UI."""

    model_config = SettingsConfigDict(
        env_prefix="AGRIK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "AgriRisk Kenya"
    environment: str = Field(default="development", description="development|staging|production")
    log_level: str = Field(default="INFO", description="Root logging level")

    # Directory layout (relative to CWD == repo root by default).
    data_root: Path = Path("data")
    models_dir: Path = Path("models")
    config_dir: Path = Path("config")

    # --- External feeds (M1) ------------------------------------------------
    # NASA Earthdata Login (free) is required to download MODIS granules.
    # Set via AGRIK_EARTHDATA_USERNAME / AGRIK_EARTHDATA_PASSWORD or .env;
    # they are never logged or committed. CHIRPS needs no credentials.
    earthdata_username: str | None = None
    earthdata_password: str | None = None
    http_timeout_s: int = Field(default=180, description="Timeout for HTTP downloads")

    # --- Derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def external_dir(self) -> Path:
        """Cache for downloaded external rasters/granules (never committed)."""
        return self.data_root / "raw" / "external"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def logs_dir(self) -> Path:
        return Path("logs")

    @property
    def pipeline_config_path(self) -> Path:
        return self.config_dir / "pipeline.yaml"

    @property
    def logging_config_path(self) -> Path:
        return self.config_dir / "logging.yaml"

    def ensure_dirs(self) -> None:
        """Create the directory skeleton if it does not yet exist."""
        for path in (
            self.raw_dir,
            self.external_dir,
            self.processed_dir,
            self.features_dir,
            self.models_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
