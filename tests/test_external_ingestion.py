"""Offline tests for the M1 external-feed layer (CHIRPS/MODIS providers).

No network access anywhere: downloads and raster files are mocked or created
locally with rasterio.
"""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
import pytest

from agrik import schemas
from agrik.config import load_pipeline_config
from agrik.ingestion import chirps, geo, modis, providers
from agrik.ingestion.base import ExternalDataError
from agrik.ingestion.raster_stats import zonal_mean
from agrik.processing import merge_datasets


@pytest.fixture(scope="module")
def rio():
    pytest.importorskip("rasterio")
    import rasterio
    import rasterio.transform

    return rasterio, rasterio.transform


# ---------------------------------------------------------------------------
# county geometry approximation
# ---------------------------------------------------------------------------
def test_county_polygons_registry_and_extent():
    polys = geo.county_polygons(50.0)
    assert len(polys) == 21
    for poly in polys.values():
        assert poly.is_valid
        left, bottom, right, top = poly.bounds
        # Kenya neighbourhood; centroid +/- ~0.5 deg for a 50 km disk.
        assert -25 < left < 60 and -25 < bottom < 25
        assert (right - left) < 1.5 and (top - bottom) < 1.2


# ---------------------------------------------------------------------------
# zonal stats over a synthetic raster (real rasterio, fake data)
# ---------------------------------------------------------------------------
def test_zonal_mean_ignores_sentinel(tmp_path, rio):
    rasterio, transform_mod = rio
    from shapely.geometry import box

    tif = tmp_path / "fake.tif"
    data = np.full((80, 75), -9999.0, dtype="float32")
    # Put 100.0 in pixels rows 20..30, cols 40..50 (lon 20..30, lat 10..20).
    data[20:30, 40:50] = 100.0
    tr = transform_mod.from_origin(-20.0, 40.0, 1.0, 1.0)  # left, top, xres, yres
    with rasterio.open(
        tif, "w", driver="GTiff", width=75, height=80, count=1,
        dtype="float32", crs="EPSG:4326", transform=tr,
    ) as dst:
        dst.write(data, 1)
    
    inside = box(20.5, 10.5, 29.5, 19.5)
    mean = zonal_mean(str(tif), inside, nodata_values=(-9999.0,))
    assert mean == pytest.approx(100.0)

    outside = box(-19.0, 38.0, -18.0, 39.0)  # all-sentinel area
    assert math.isnan(zonal_mean(str(tif), outside, nodata_values=(-9999.0,)))

    # scale applied after validity filter (MODIS-style 1e-4 integers)
    scaled = zonal_mean(str(tif), inside, valid_range=(0, 2000), scale=1e-4)
    assert scaled == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# CHIRPS connector (mocked download; real rasterio aggregation)
# ---------------------------------------------------------------------------
def test_chirps_month_url():
    assert (
        chirps.month_url("https://base/tifs/", 2020, 1)
        == "https://base/tifs/chirps-v2.0.2020.01.tif.gz"
    )


def _config_single_year():
    cfg = deepcopy(load_pipeline_config())
    cfg["panel"]["start_year"] = 2019
    cfg["panel"]["end_year"] = 2019
    return cfg


def test_chirps_build_panel_mocked(tmp_path, rio, monkeypatch):
    rasterio, transform_mod = rio
    cfg = _config_single_year()
    from agrik.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    get_settings.cache_clear()

    def fake_ensure(base_url, year, month, cache_dir, *, timeout):
        tif = cache_dir / f"fake-{year}-{month:02d}.tif"
        tif.parent.mkdir(parents=True, exist_ok=True)
        data = np.full((80, 75), 30.0 + month, dtype="float32")
        tr = transform_mod.from_origin(-20.0, 40.0, 1.0, 1.0)
        with rasterio.open(
            tif, "w", driver="GTiff", width=75, height=80, count=1,
            dtype="float32", crs="EPSG:4326", transform=tr,
        ) as dst:
            dst.write(data, 1)
        return tif

    monkeypatch.setattr(chirps, "ensure_month_tif", fake_ensure)
    df = chirps.build_climate_panel(config=cfg)

    assert len(df) == 21 * 12
    assert set(df[schemas.PROVENANCE_COLUMN].unique()) == {"chirps"}
    assert "temp_mean_c" not in df.columns  # omitted, not faked
    # January value 31.0 everywhere (constant raster -> zonal mean exact).
    jan = df[df["month"] == 1]["precipitation_mm"]
    assert jan.astype(float).mean() == pytest.approx(31.0, abs=0.05)


def test_chirps_all_months_failed_raises(tmp_path, monkeypatch):
    cfg = _config_single_year()
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    from agrik.settings import get_settings

    get_settings.cache_clear()

    def boom(*a, **k):
        raise OSError("network down")

    monkeypatch.setattr(chirps, "ensure_month_tif", boom)
    with pytest.raises(ExternalDataError):
        chirps.build_climate_panel(config=cfg)


# ---------------------------------------------------------------------------
# provider registry & fallback policy
# ---------------------------------------------------------------------------
def test_registry_lists_real_sources():
    assert providers.available_sources() == ["chirps", "modis"]


def test_unknown_source_rejected():
    cfg = deepcopy(load_pipeline_config())
    cfg["datasets"]["climate"]["source"] = "weatherball"
    with pytest.raises(ValueError, match="unknown source"):
        providers.ingest_datasets(config=cfg)


def test_provider_failure_falls_back_labelled(monkeypatch, tmp_path):
    cfg = deepcopy(load_pipeline_config())
    cfg["datasets"]["climate"]["source"] = "chirps"
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    from agrik.settings import get_settings

    get_settings.cache_clear()

    def boom(config=None, settings=None):
        raise ExternalDataError("simulated outage")

    monkeypatch.setattr(providers.chirps, "build_climate_panel", boom)
    frames = providers.ingest_datasets(config=cfg)
    assert set(frames["climate"][schemas.PROVENANCE_COLUMN].unique()) == {"synthetic"}

    cfg["external"]["allow_synthetic_fallback"] = False
    with pytest.raises(ExternalDataError):
        providers.ingest_datasets(config=cfg)


def test_merge_keeps_per_dataset_sources():
    keys = {
        "county_code": ["001", "001"],
        "county_name": ["Bungoma", "Bungoma"],
        "year": [2020, 2020],
        "month": [1, 2],
        "date": ["2020-01-01", "2020-02-01"],
    }
    climate = pd.DataFrame({**keys, "precipitation_mm": [10.0, 20.0],
                            schemas.PROVENANCE_COLUMN: "chirps"})
    market = pd.DataFrame({**keys, "maize_price_kes_kg": [100.0, 110.0],
                           schemas.PROVENANCE_COLUMN: "synthetic"})
    merged = merge_datasets({"climate": climate, "market": market})
    assert merged["climate_source"].eq("chirps").all()
    assert merged["market_source"].eq("synthetic").all()
    assert set(merged[schemas.PROVENANCE_COLUMN].unique()) == {"mixed"}


# ---------------------------------------------------------------------------
# MODIS pure helpers (no network, no HDF needed)
# ---------------------------------------------------------------------------
def test_modis_composite_dates():
    start = modis.composite_start("MOD13A1.A2020049.h26v10.061.2020065001234.hdf")
    assert start == date(2020, 2, 18)
    assert modis.composite_month(start) == (2020, 2)  # midpoint ~2020-02-26
    assert modis.composite_start("bogus") is None


def test_modis_pick_download_url():
    links = [
        {"href": "https://cmr.earthdata.nasa.gov/meta.json"},
        {"href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/"
                 "MOD13A1.061/MOD13A1.A2020049.h26v10.061.2020065001234.hdf"},
    ]
    url = modis.pick_download_url(links)
    assert url and url.endswith(".hdf") and "earthdatacloud" in url


def test_modis_granule_intersects_county():
    polys = geo.county_polygons(50.0)  # real county disks around centroids
    # A box over Kenya (CMR "S W N E" order) must intersect some county disk.
    kenya_granule = {"boxes": ["-1.5 36.5 1.0 38.5"]}
    assert modis.granule_intersects(kenya_granule, polys) is True
    # A box in the South Atlantic intersects nothing (and no boxes -> kept).
    ocean_granule = {"boxes": ["-40.0 -20.0 -38.0 -18.0"]}
    assert modis.granule_intersects(ocean_granule, polys) is False
    assert modis.granule_intersects({"boxes": []}, polys) is True


def test_modis_requires_credentials(monkeypatch, tmp_path):
    import requests

    monkeypatch.setattr(requests.utils, "get_netrc_auth", lambda *a, **k: None)
    from agrik.settings import Settings

    with pytest.raises(ExternalDataError, match="Earthdata"):
        modis.earthdata_session(Settings(earthdata_username=None,
                                         earthdata_password=None))
