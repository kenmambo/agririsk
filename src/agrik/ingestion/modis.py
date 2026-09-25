"""MODIS MOD13A1 v6.1 vegetation-index provider (real data, needs credentials).

Dataset produced: ``vegetation`` (county-monthly), variables ``ndvi`` and
``evi`` - monthly means of the 16-day 500 m MODIS composites over each county
approximation (see :mod:`agrik/ingestion/geo`).

Access requirements
-------------------
* Granule *discovery* uses NASA CMR, which is public (no auth).
* Granule *download* requires a free NASA Earthdata Login account. Provide it
  either via ``AGRIK_EARTHDATA_USERNAME`` / ``AGRIK_EARTHDATA_PASSWORD``
  (or ``.env``) or a standard ``~/.netrc`` entry for
  ``machineursdauth.earthdata.nasa.gov`` / ``machine data.lpdaac...``.
  Credentials are never logged or written to disk by this package.

Provenance: every row is stamped ``data_source = "modis"``. If credentials or
the network are unavailable this connector raises
:class:`~agrik.ingestion.base.ExternalDataError`; the orchestration layer then
either aborts or falls back to the clearly-labelled synthetic slice.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .geo import county_polygons
from .raster_stats import zonal_mean
from .utils import download_cached

LOGGER = get_logger("ingestion.modis")

SOURCE_NAME = "modis"
CMR_BASE = "https://cmr.earthdata.nasa.gov/search"
_COMPOSITE_DAYS = 16
_TITLE_RE = re.compile(r"\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.")


def resolve_collection(session, short_name: str, version: str) -> str:
    """CMR concept id for a collection (public search API)."""
    resp = session.get(
        f"{CMR_BASE}/collections.json",
        params={"short_name": short_name, "version": version},
        timeout=30,
    )
    resp.raise_for_status()
    entries = resp.json()["feed"]["entry"]
    if not entries:
        raise ExternalDataError(f"CMR: no collection {short_name} v{version}.")
    # CMR's collections.json exposes the concept id under "id" (older format: "concept-id").
    return entries[0].get("id") or entries[0]["concept-id"]


def search_granules(
    session, concept_id: str, bbox: list[float], start_year: int, end_year: int
) -> list[dict[str, Any]]:
    """CMR granule search for the Kenya bbox over the panel window (public)."""
    west, south, east, north = bbox
    temporal = (
        f"{start_year}-01-01T00:00:00Z,"
        f"{date(end_year, 12, 31).isoformat()}T23:59:59Z"
    )
    resp = session.get(
        f"{CMR_BASE}/granules.json",
        params={
            "concept_id": concept_id,
            "bounding_box": f"{west},{south},{east},{north}",
            "temporal": temporal,
            "page_size": 2000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    entries = resp.json()["feed"]["entry"]
    out = []
    for e in entries:
        url = pick_download_url(e.get("online-accesses", e.get("links", [])))
        if url:
            out.append(
                {
                    "title": e.get("title", url),
                    "time_start": e["time_start"],
                    "url": url,
                    "boxes": e.get("boxes", []),
                }
            )
    return out


def granule_intersects(granule: dict[str, Any], polygons: dict[str, object]) -> bool:
    """Cheap client-side filter: granule geographic box vs any county disk."""
    from shapely.geometry import box

    geoms = []
    for b in granule.get("boxes", []):
        try:
            s, w, n, e_ = (float(x) for x in b.split())
            geoms.append(box(w, s, e_, n))
        except ValueError:
            continue
    if not geoms:  # no simplified box -> keep (search bbox already filtered)
        return True
    return any(g.intersects(p) for g in geoms for p in polygons.values())


def pick_download_url(links: list[dict]) -> str | None:
    """Direct S3/HTTPS granule URL (skip metadata / license links)."""
    candidates = [
        link["href"] for link in links
        if link.get("href", "").lower().endswith(".hdf") and link["href"].startswith("https://")
    ]
    for c in candidates:
        if "earthdatacloud" in c:
            return c
    return candidates[0] if candidates else None


def composite_start(title_or_iso: str) -> date | None:
    """Granule composite start date from title ``...AYYYYDDD...`` (or time_start ISO)."""
    m = _TITLE_RE.search(title_or_iso)
    if m:
        year, doy = int(m.group(1)), int(m.group(2))
        return (datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)).date()
    try:  # fall back to an ISO timestamp
        return datetime.fromisoformat(title_or_iso.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def composite_month(start: date, duration_days: int = _COMPOSITE_DAYS) -> tuple[int, int]:
    """(year, month) of the composite *midpoint* - its nominal reporting month."""
    mid = start + timedelta(days=duration_days // 2)
    return mid.year, mid.month


def earthdata_session(settings: Settings):
    """Authenticated session for granule download (or clear error)."""
    import requests

    sess = requests.Session()
    if settings.earthdata_username and settings.earthdata_password:
        sess.auth = (settings.earthdata_username, settings.earthdata_password)
        return sess
    auth = requests.utils.get_netrc_auth(
        "https://data.lpdaac.earthdatacloud.nasa.gov/", raise_errors=False
    )
    if auth:  # requests applies ~/.netrc automatically for matching hosts
        return sess
    raise ExternalDataError(
        "MODIS needs a NASA Earthdata Login (free at https://urs.earthdata.nasa.gov). "
        "Set AGRIK_EARTHDATA_USERNAME/AGRIK_EARTHDATA_PASSWORD (or .env) or a ~/.netrc entry."
    )


def hdf_band_paths(hdf_path) -> dict[str, str]:
    """Map 'ndvi'/'evi' -> GDAL HDF4 subdataset names inside a MOD13A1 granule."""
    import rasterio

    wanted: dict[str, str] = {}
    for sub in rasterio.subdatasets(str(hdf_path)):
        low = sub.lower()
        if "ndvi" in low and "vegetation" in low:
            wanted["ndvi"] = sub
        elif "evi" in low and "vegetation" in low:
            wanted["evi"] = sub
    if "ndvi" not in wanted or "evi" not in wanted:
        raise ExternalDataError(f"MODIS granule {hdf_path} missing NDVI/EVI bands.")
    return wanted


def build_vegetation_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Download covering MODIS granules and aggregate to the county-month schema."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["modis"]
    panel = config["panel"]
    start_year, end_year = int(panel["start_year"]), int(panel["end_year"])

    polys = county_polygons(float(config["external"].get("county_buffer_km", 50)))
    names = {c.code: c.name for c in counties.all_counties()}
    cache_dir = settings.external_dir / "modis"

    sess = earthdata_session(settings)
    concept = resolve_collection(sess, ext["cmr_short_name"], str(ext["version"]))
    granules = search_granules(sess, concept, ext["bbox"], start_year, end_year)
    if not granules:
        raise ExternalDataError("MODIS: CMR returned no granules for Kenya window.")
    before = len(granules)
    granules = [g for g in granules if granule_intersects(g, polys)]
    if before != len(granules):
        LOGGER.info("MODIS: dropped %d/%d granules not intersecting any county.",
                    before - len(granules), before)
    if not granules:
        raise ExternalDataError("MODIS: no granules intersect the county approximations.")

    max_downloads = int(ext.get("max_granule_downloads", 60))
    if len(granules) > max_downloads:
        LOGGER.warning(
            "MODIS: %d granules cover the window; sampling %d evenly across "
            "time (volume guard). Coverage will be partial - raise "
            "external.modis.max_granule_downloads or narrow the panel window.",
            len(granules), max_downloads,
        )
        idx = np.linspace(0, len(granules) - 1, max_downloads).round().astype(int)
        granules = [granules[int(i)] for i in sorted(set(idx.tolist()))]

    # samples[(county, year, month)] -> list of band means
    samples: dict[tuple[str, int, int], dict[str, list[float]]] = {}
    for g in granules:
        cstart = composite_start(g["title"]) or composite_start(g["time_start"])
        if cstart is None:
            continue
        ym = composite_month(cstart)
        if not (start_year <= ym[0] <= end_year):
            continue
        hdf = download_cached(g["url"], cache_dir / g["url"].rsplit("/", 1)[-1],
                              timeout=settings.http_timeout_s, session=sess)
        try:
            bands = hdf_band_paths(hdf)
        except ExternalDataError as exc:
            LOGGER.warning("skipping granule %s: %s", g["title"], exc)
            continue
        for code, poly in polys.items():
            rec = samples.setdefault((code, ym[0], ym[1]), {"ndvi": [], "evi": []})
            for var, sub in bands.items():
                val = zonal_mean(
                    sub, poly, valid_range=(-2000, 10000), scale=1e-4,
                )
                if not (isinstance(val, float) and math.isnan(val)):
                    rec[var].append(val)

    rows = []
    for (code, year, month), rec in sorted(samples.items()):
        rows.append(
            {
                "county_code": code,
                schemas.COUNTY_NAME_COLUMN: names[code],
                "year": year,
                "month": month,
                schemas.DATE_COLUMN: date(year, month, 1).isoformat(),
                "ndvi": round(sum(rec["ndvi"]) / len(rec["ndvi"]), 4) if rec["ndvi"] else None,
                "evi": round(sum(rec["evi"]) / len(rec["evi"]), 4) if rec["evi"] else None,
                schemas.PROVENANCE_COLUMN: SOURCE_NAME,
            }
        )
    if not rows:
        raise ExternalDataError("MODIS: no granules could be aggregated.")
    LOGGER.info("MODIS vegetation panel: %d rows (REAL data).", len(rows))
    return pd.DataFrame(rows)
