"""MODIS MOD13A1 v6.1 vegetation-index provider (real data, needs credentials).

Dataset produced: ``vegetation`` (county-monthly), variables ``ndvi`` and
``evi`` - monthly means of the 16-day 500 m MODIS composites over each county
approximation (see :mod:`agrik/ingestion/geo`).

Access requirements
-------------------
* Granule *discovery* uses NASA CMR, which is public (no auth).
* Granule *download* requires a free NASA Earthdata Login. Preferred:
  an application token (``AGRIK_EARTHDATA_TOKEN``, sent as a Bearer header -
  works with MFA/SSO accounts); fallback: username + password Basic auth or a
  ``~/.netrc`` entry. Credentials are never logged or committed.
* Granules are HDF4; the rasterio pip wheel lacks the HDF4 driver, so bands
  are read with ``pyhdf`` and georeferenced from each granule's own
  ``StructMetadata.0`` sinusoidal grid definition (pyproj + shapely zonal).

Provenance: every row is stamped ``data_source = "modis"``. If credentials or
the network are unavailable this connector raises
:class:`~agrik.ingestion.base.ExternalDataError`; the orchestration layer then
either aborts or falls back to the clearly-labelled synthetic slice.
"""

from __future__ import annotations

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
from .utils import download_cached

LOGGER = get_logger("ingestion.modis")

SOURCE_NAME = "modis"
CMR_BASE = "https://cmr.earthdata.nasa.gov/search"
_COMPOSITE_DAYS = 16
_TITLE_RE = re.compile(r"\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.")
# MOD13A1 v6.1 SDS names (spaces, not underscores) for the 500 m products.
_SDS_FOR_VAR = {"ndvi": "500m 16 days NDVI", "evi": "500m 16 days EVI"}


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


class TokenAuth:
    """requests auth handler attaching an Earthdata app token as ``Bearer``.

    Implemented as an AuthBase callable (not a plain session header) so the
    header is re-applied after every redirect - NASA granule downloads bounce
    between cmr.earthdata.nasa.gov, urs.earthdata.nasa.gov and the LP DAAC
    host, and requests strips Authorization headers on cross-host redirects.
    """

    def __init__(self, token: str) -> None:
        self.token = token

    def __call__(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


def earthdata_session(settings: Settings):
    """Authenticated session for granule download (or clear error).

    Preference order:
    1. Application token (urs.earthdata.nasa.gov -> Profile -> Applications
       -> Generate Token) sent as a Bearer header; works with MFA/SSO accounts.
    2. Username + password Basic auth (fails for MFA/IdP-only accounts).
    3. A ``~/.netrc`` entry for the LP DAAC host.
    """
    import requests

    sess = requests.Session()
    if settings.earthdata_token:
        sess.auth = TokenAuth(settings.earthdata_token)
        return sess
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
        "Set AGRIK_EARTHDATA_TOKEN (recommended: urs.earthdata.nasa.gov -> Applications "
        "-> Generate Token), or AGRIK_EARTHDATA_USERNAME/PASSWORD, or a ~/.netrc entry."
    )


def hdf_band_paths(hdf_path) -> dict[str, str]:
    """Map 'ndvi'/'evi' -> MOD13A1 SDS names inside a granule (via pyhdf).

    The rasterio pip wheel ships without the GDAL HDF4 driver, so MOD13A1
    (.hdf = HDF4) is read directly with pyhdf; only the *names* of the two
    500 m vegetation bands are resolved here.
    """
    from pyhdf.SD import SD, SDC

    sds_file = SD(str(hdf_path), SDC.READ)
    try:
        available = set(sds_file.datasets())
    finally:
        sds_file.end()
    wanted = {var: name for var, name in _SDS_FOR_VAR.items() if name in available}
    if not wanted:
        raise ExternalDataError(f"MODIS granule {hdf_path} has no 500m NDVI/EVI SDS.")
    return wanted


def parse_grid_extent(struct_metadata: str) -> tuple[float, float, float, float, float]:
    """(ulx, uly, lrx, lry, radius_m) sinusoidal grid extent from StructMetadata.0.

    MOD13A1 granules self-describe their tile rectangle in projected metres,
    e.g. ``UpperLeftPointMtrs=(3335851.559,1111950.5197)`` for h21v08.
    """
    ul = re.search(r"UpperLeftPointMtrs=\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
                   struct_metadata)
    lr = re.search(r"LowerRightMtrs=\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)",
                   struct_metadata)
    rr = re.search(r"ProjParams=\(\s*([-\d.eE+]+)", struct_metadata)
    if not (ul and lr):
        raise ExternalDataError("MODIS granule: no grid extent in StructMetadata.0")
    radius = float(rr.group(1)) if rr else 6371007.181
    return (float(ul.group(1)), float(ul.group(2)),
            float(lr.group(1)), float(lr.group(2)), radius)


def granule_county_means(hdf_path, polygons: dict[str, object]) -> dict[str, dict[str, float]]:
    """Zonal mean NDVI/EVI per county polygon for one granule (REAL pixels).

    Pixel centres are placed on the sinusoidal grid defined by the granule's
    own extent, inverse-projected to lon/lat (pyproj) and point-in-polygon
    filtered (shapely). Raw integers are validity-filtered (fill value and
    product valid_range) before applying the scale factor.
    """
    import numpy as np
    import shapely
    from pyhdf.SD import SD, SDC
    from pyproj import CRS, Transformer

    sds_file = SD(str(hdf_path), SDC.READ)
    try:
        ulx, uly, lrx, lry, radius = parse_grid_extent(
            str(sds_file.attributes().get("StructMetadata.0", ""))
        )
        arrays: dict[str, np.ndarray] = {}
        scale: dict[str, float] = {}
        meta: dict[str, tuple[float, object]] = {}
        for var, sds_name in _SDS_FOR_VAR.items():
            if sds_name not in sds_file.datasets():
                continue
            sd = sds_file.select(sds_name)
            attrs = sd.attributes()
            arrays[var] = np.asarray(sd.get(), dtype=np.float64)
            scale[var] = 1.0 / float(attrs.get("scale_factor", 10000.0))
            meta[var] = (attrs.get("_FillValue"), attrs.get("valid_range"))
        if not arrays:
            raise ExternalDataError(f"MODIS granule {hdf_path}: no NDVI/EVI SDS found.")

        ny, nx = next(iter(arrays.values())).shape
        xres, yres = (lrx - ulx) / nx, (uly - lry) / ny
        sinu = CRS.from_proj4(f"+proj=sinu +R={radius} +nadgrids=@null +wktext")
        to_proj = Transformer.from_crs("EPSG:4326", sinu, always_xy=True)
        to_geo = Transformer.from_crs(sinu, "EPSG:4326", always_xy=True)

        out: dict[str, dict[str, float]] = {}
        for code, poly in polygons.items():
            minx, miny, maxx, maxy = to_proj.transform_bounds(*poly.bounds, densify_pts=21)
            c0 = max(int((minx - ulx) / xres) - 1, 0)
            c1 = min(int((maxx - ulx) / xres) + 2, nx)
            r0 = max(int((uly - maxy) / yres) - 1, 0)  # rows run north -> south
            r1 = min(int((uly - miny) / yres) + 2, ny)
            if c0 >= c1 or r0 >= r1:
                continue  # county disk does not touch this tile
            xs = ulx + (np.arange(c0, c1) + 0.5) * xres
            ys = uly - (np.arange(r0, r1) + 0.5) * yres
            gx, gy = np.meshgrid(xs, ys)
            lons, lats = to_geo.transform(gx.ravel(), gy.ravel())
            inside = shapely.contains(
                poly, shapely.points(np.column_stack([lons, lats]))
            ).reshape(gx.shape)
            if not inside.any():
                continue
            rec: dict[str, float] = {}
            for var, arr in arrays.items():
                fill, valid = meta[var]
                sub = arr[r0:r1, c0:c1][inside]
                keep = np.isfinite(sub)
                if fill is not None:
                    keep &= sub != float(np.asarray(fill).ravel()[0])
                if valid is not None:
                    v = np.asarray(valid, dtype=np.float64).ravel()
                    keep &= (sub >= v[0]) & (sub <= v[1])
                vals = sub[keep] * scale[var]
                if vals.size:
                    rec[var] = float(vals.mean())
            if rec:
                out[code] = rec
        return out
    finally:
        sds_file.end()  # pyhdf: close the SDFile itself (SDS handles need none)


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
            means = granule_county_means(hdf, polys)
        except ExternalDataError as exc:
            LOGGER.warning("skipping granule %s: %s", g["title"], exc)
            continue
        for code, rec in means.items():
            bucket = samples.setdefault((code, ym[0], ym[1]), {"ndvi": [], "evi": []})
            for var, val in rec.items():
                bucket[var].append(val)

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
