"""Tests for the feature-engineering pipeline (incl. no-lookahead guarantee)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import agrik.schemas as schemas
from agrik.features import build_features, engineered_feature_names, make_xy
from agrik.ingestion.synthetic import build_master_panel


def test_build_features_adds_expected_columns():
    master = build_master_panel(2019, 2021, seed=5)
    feat = build_features(master)
    for name in engineered_feature_names():
        assert name in feat.columns


def test_lag_uses_only_past_information():
    master = build_master_panel(2019, 2021, seed=5)
    feat = build_features(master)
    sub = feat[feat["county_code"] == "001"].sort_values(["year", "month"])
    # First lagged observation is undefined (no earlier month) ...
    assert pd.isna(sub["precip_lag1"].iloc[0])
    # ... and every later lagged value equals the previous month's rainfall.
    assert np.allclose(sub["precip_lag1"].iloc[1:].to_numpy(),
                       sub["precipitation_mm"].iloc[:-1].to_numpy())


def test_changing_future_value_does_not_alter_past_features():
    master = build_master_panel(2019, 2020, seed=5)
    base = build_features(master)
    modified = master.copy()
    last = modified.index[modified["county_code"] == "001"][-1]
    modified.loc[last, "precipitation_mm"] = 999.0  # tamper with the final month
    after = build_features(modified)
    first_rows = after[after["county_code"] == "001"].iloc[:-1]["precip_roll3"]
    base_rows = base[base["county_code"] == "001"].iloc[:-1]["precip_roll3"]
    pd.testing.assert_series_equal(first_rows.reset_index(drop=True),
                                   base_rows.reset_index(drop=True))


def test_make_xy_excludes_target_and_meta():
    master = build_master_panel(2019, 2021, seed=5)
    feat = build_features(master)
    X, y, feats = make_xy(feat)
    assert schemas.TARGET not in feats
    assert schemas.TARGET not in X.columns
    assert len(X) == len(y)
    assert X.notna().all().all()  # warm-up NaN rows dropped
