"""Open-Meteo ERA5 temperature connector - REAL data.

Supplies the ``climate.temp_mean_c`` variable that CHIRPS does not carry:
daily mean 2 m air temperature from the ERA5 reanalysis (Open-Meteo
Archive API, free, no auth), averaged to calendar months in-code (no
opaque server-side monthly aggregate).

Design: one request PER COUNTY covering the whole panel window (21 calls,
~18 KB each), cached as raw JSON under ``data/raw/external/openmeteo/`` so
re-runs are offline. Temperature is a slow-varying signal - a single point
at the county registry centroid is defensible (unlike buffered rainfall).

Provenance: enriched rows carry ``data_source = "chirps+openmeteo"``.
Counties/months with no temperature stay NULL (never faked) and are
logged; :func:`merge_temperature` leaves the CHIRPS frame untouched when
nothing can be added.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .utils import download_cached

LOGGER = get_logger("ingestion.openmeteo")

SOURCE_NAME = "openmeteo"
COMBINED_SOURCE = "chirps+openmeteo"
TEMPERATURE_COLUMN = "temp_mean_c"

# ERA5 is a 0.25 deg grid: two centroids closer than ~0.25 deg would receive
# an identical series - logged as a warning because per-county contrast then
# does not exist for that pair.
_MIN_DISTINCT_DEG = 0.25


def api_url(base_url: str, lat: float, lon: float, sy: int, ey: int) -> str:
    """One daily-temperature request covering the panel window."""
    return (
        f"{base_url}?latitude={lat}&longitude={lon}"
        f"&start_date={sy}-01-01&end_date={ey}-12-31"
        f"&daily=temperature_2m_mean&timezone=UTC"
    )


def parse_daily_temps(payload: dict) -> pd.DataFrame:
    """Open-Meteo response -> DataFrame[county_year, county_month, temp_mean_c].

    Null days (ERA5 gap) are dropped; a county whose series is entirely
    null raises so the caller can keep the CHIRPS-only frame.
    """
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    rows = [
        (t[:4], t[5:7], float(v))
        for t, v in zip(times, temps, strict=False)  # API pairs them 1:1
        if t is not None and v is not None
    ]
    if not rows:
        raise ExternalDataError("Open-Meteo: response has no non-null daily temperatures.")
    return pd.DataFrame(rows, columns=["y", "m", "temp_mean_c"])


def build_temperature_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """County-monthly mean temperature (degC) from ERA5 daily means."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["openmeteo"]
    base_url = str(ext["base_url"])
    panel = config["panel"]
    sy, ey = int(panel["start_year"]), int(panel["end_year"])
    cache = settings.external_dir / "openmeteo"

    registry = counties.all_counties()
    by_name = {c.name: c for c in registry}
    seen: dict[tuple[float, float], str] = {}
    frames = []
    for c in registry:
        lat, lon = round(c.lat, 3), round(c.lon, 3)
        key = (lat, lon)
        if key in seen:
            other = by_name[seen[key]]
            gap = max(abs(c.lat - other.lat), abs(c.lon - other.lon))
            if gap < _MIN_DISTINCT_DEG:
                LOGGER.warning(
                    "Open-Meteo: %s and %s fall in the same ERA5 grid cell "
                    "(%.2f deg apart) - identical series; per-county temperature "
                    "contrast between them is meaningless.", c.name, other.name, gap,
                )
        else:
            seen[key] = c.name
        dest = cache / f"temps_{lat}_{lon}_{sy}_{ey}.json"
        try:
            raw = download_cached(
                api_url(base_url, lat, lon, sy, ey), dest, timeout=settings.http_timeout_s
            )
            daily = parse_daily_temps(json.loads(raw.read_text(encoding="utf-8")))
        except Exception as exc:  # per-county failure: skip loudly, keep others
            LOGGER.warning("Open-Meteo %s (%s): %s", c.name, key, exc)
            continue
        daily["county_code"] = c.code
        frames.append(daily)

    if not frames:
        raise ExternalDataError("Open-Meteo: no county temperature series could be fetched.")
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["y"].astype(int)
    df["month"] = df["m"].astype(int)
    monthly = (
        df.groupby(["county_code", "year", "month"])[TEMPERATURE_COLUMN]
        .mean()
        .round(2)
        .reset_index()
    )
    missing = {c.code for c in registry} - set(monthly["county_code"])
    if missing:
        LOGGER.warning(
            "Open-Meteo: no temperature for %d counties (their temp_mean_c stays "
            "NULL, not faked): %s",
            len(missing), ", ".join(sorted(missing)),
        )
    LOGGER.info(
        "Open-Meteo temperature: %d county-months of REAL ERA5 daily means "
        "aggregated to months (unit: degC).", len(monthly),
    )
    return monthly[["county_code", "year", "month", TEMPERATURE_COLUMN]]


def merge_temperature(climate: pd.DataFrame, temps: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``temp_mean_c`` into a CHIRPS climate frame (provenance honest).

    Only rows whose temperature value THIS join contributed are re-stamped
    ``chirps+openmeteo``. Rows that already carried a value (e.g. a synthetic
    frame) keep their original ``data_source`` - attribution never lies - and
    unmatched rows keep a NULL ``temp_mean_c`` (never interpolated here).
    """
    already_had = TEMPERATURE_COLUMN in climate.columns
    if already_had:
        return climate  # nothing to attribute to Open-Meteo; do not double-join
    out = climate.merge(temps, on=["county_code", "year", "month"], how="left")
    got = out[TEMPERATURE_COLUMN].notna()
    out.loc[got, schemas.PROVENANCE_COLUMN] = COMBINED_SOURCE
    n_missing = int((~got).sum())
    if n_missing:
        LOGGER.warning(
            "Open-Meteo merge: %d/%d climate rows have no temperature "
            "(NULL kept, not faked).", n_missing, len(out),
        )
    return out
