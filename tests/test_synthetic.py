"""Tests for the synthetic data generator (provenance + integrity)."""

from __future__ import annotations

import math

import agrik.schemas as schemas
from agrik.ingestion.synthetic import build_master_panel, build_synthetic_datasets


def test_master_panel_integrity():
    df = build_master_panel(2020, 2021, seed=1)
    for col in ["county_code", "year", "month", "precipitation_mm", "ndvi",
                "maize_price_kes_kg", schemas.TARGET, schemas.PROVENANCE_COLUMN]:
        assert col in df.columns
    # Every row is explicitly marked synthetic.
    assert set(df[schemas.PROVENANCE_COLUMN].unique()) == {"synthetic"}
    # One row per county-month (no duplicates), target bounded 0-100.
    assert not df.duplicated(["county_code", "year", "month"]).any()
    assert df[schemas.TARGET].between(0, 100).all()
    assert (df["precipitation_mm"] >= 0).all()


def test_datasets_split_by_schema():
    ds = build_synthetic_datasets(seed=2)
    assert set(ds) >= set(schemas.DATASET_VARIABLES)
    assert "ndvi" in ds["vegetation"].columns
    assert "precipitation_mm" in ds["climate"].columns
    # Outcome dataset carries the model target.
    assert schemas.TARGET in ds["outcome"].columns


def test_deterministic_with_seed():
    a = build_master_panel(2020, 2020, seed=7)
    b = build_master_panel(2020, 2020, seed=7)
    assert math.isclose(a[schemas.TARGET].sum(), b[schemas.TARGET].sum())
