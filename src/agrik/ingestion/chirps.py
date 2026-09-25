"""CHIRPS v2.0 monthly rainfall provider (real data, no authentication).

Dataset produced: ``climate`` (county-monthly), variables:

- ``precipitation_mm`` : mean CHIRPS monthly rainfall over each county
  approximation (see :mod:`agrik.ingestion.geo`).

``temp_mean_c`` is **not** provided by this feed and is deliberately omitted
rather than faked - downstream feature engineering adapts to the columns that
exist. A temperature feed (e.g. ERA5 via Open-Meteo) is a future connector.

Provenance: every row is stamped ``data_source = "chirps"``.

Product reference: CHIRPS v2.0 (Climate Hazards center Infra-Run Rainfall with
Stations, UCSB Climate Hazards Center) - Africa monthly composite, 0.05 deg,
gzip GeoTIFF, fill sentinel -9999 (not flagged in the header).
"""

from __future__ import annotations

from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .geo import county_polygons
from .raster_stats import zonal_mean
from .utils import download_cached, ensure_gunzip

LOGGER = get_logger("ingestion.chirps")

SOURCE_NAME = "chirps"


def month_url(base_url: str, year: int, month: int) -> str:
    """URL of the Africa monthly CHIRPS composite for ``year-month``."""
    return f"{base_url}chirps-v2.0.{year}.{month:02d}.tif.gz"


def ensure_month_tif(
    base_url: str, year: int, month: int, cache_dir: Path, *, timeout: int
) -> Path:
    """Download (cached) and unpack one monthly composite; returns the .tif path."""
    gz = cache_dir / f"chirps-v2.0.{year}.{month:02d}.tif.gz"
    download_cached(month_url(base_url, year, month), gz, timeout=timeout)
    return ensure_gunzip(gz)


def _panel_months(config: dict[str, Any]) -> list[tuple[int, int]]:
    panel = config["panel"]
    years = range(int(panel["start_year"]), int(panel["end_year"]) + 1)
    return [(y, m) for y, m in product(years, range(1, 13))]


def build_climate_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Aggregate real CHIRPS monthly rainfall into the ``climate`` schema.

    Raises :class:`ExternalDataError` if *no* month could be fetched, so the
    caller can decide about fallback; partial coverage is logged loudly and
    returned as-is (missing months stay gaps, filled only in processing).
    """
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["chirps"]
    base_url = str(ext["base_url"])
    nodata = float(ext.get("nodata", -9999))
    cache_dir = settings.external_dir / "chirps"

    polys = county_polygons(float(config["external"].get("county_buffer_km", 50)))
    names = {c.code: c.name for c in counties.all_counties()}

    rows: list[dict[str, Any]] = []
    ok_months: set[tuple[int, int]] = set()
    failed: list[tuple[int, int]] = []
    for year, month in _panel_months(config):
        try:
            tif = ensure_month_tif(
                base_url, year, month, cache_dir, timeout=settings.http_timeout_s
            )
        except Exception as exc:  # network / HTTP error for this month
            LOGGER.warning("CHIRPS %04d-%02d unavailable: %s", year, month, exc)
            failed.append((year, month))
            continue
        for code, poly in polys.items():
            mean_mm = zonal_mean(
                str(tif), poly, nodata_values=(nodata,), valid_range=(0.0, 2000.0)
            )
            rows.append(
                {
                    "county_code": code,
                    schemas.COUNTY_NAME_COLUMN: names[code],
                    "year": year,
                    "month": month,
                    schemas.DATE_COLUMN: date(year, month, 1).isoformat(),
                    "precipitation_mm": None if pd.isna(mean_mm) else round(mean_mm, 2),
                    schemas.PROVENANCE_COLUMN: SOURCE_NAME,
                }
            )
        ok_months.add((year, month))

    if not ok_months:
        raise ExternalDataError(
            f"CHIRPS: could not fetch any of {len(_panel_months(config))} months."
        )
    if failed:
        LOGGER.warning(
            "CHIRPS partial coverage: %d/%d months fetched; missing: %s",
            len(ok_months), len(_panel_months(config)),
            ", ".join(f"{y}-{m:02d}" for y, m in failed),
        )
    df = pd.DataFrame(rows)
    n_empty = int(df["precipitation_mm"].isna().sum())
    LOGGER.info(
        "CHIRPS climate panel: %s rows, %d months (REAL data; %d empty county-cells).",
        len(df), len(ok_months), n_empty,
    )
    return df
