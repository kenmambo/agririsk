"""Tests for the preprocessing primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agrik.processing.preprocess import clean_dataset, impute_panel, merge_datasets


def test_clean_coerces_code_and_dedupes():
    df = pd.DataFrame(
        {
            "county_code": [1, "001"],  # both normalise to "001"
            "year": [2020, 2020],
            "month": [1, 1],
            "ndvi": [0.5, 0.9],
        }
    )
    out = clean_dataset(df)
    assert len(out) == 1  # duplicate grain dropped (keep first)
    assert out["county_code"].iloc[0] == "001"
    assert np.isclose(out["ndvi"].iloc[0], 0.5)


def test_clean_coerces_non_numeric_to_nan():
    df = pd.DataFrame(
        {
            "county_code": ["001"],
            "year": [2020],
            "month": [1],
            "ndvi": ["not-a-number"],
        }
    )
    out = clean_dataset(df)
    assert pd.isna(out["ndvi"].iloc[0])


def test_impute_interpolates_interior_gaps():
    df = pd.DataFrame(
        {
            "county_code": ["001", "001", "001"],
            "year": [2020, 2020, 2020],
            "month": [1, 2, 3],
            "ndvi": [0.2, np.nan, 0.6],
        }
    )
    out = impute_panel(df, columns=["ndvi"])
    assert np.isclose(out["ndvi"].iloc[1], 0.4)


def test_merge_datasets_joins_on_grain():
    climate = pd.DataFrame(
        {"county_code": ["001", "001"], "county_name": ["Mombasa", "Mombasa"],
         "year": [2020, 2021], "month": [1, 1], "precipitation_mm": [50.0, 60.0]}
    )
    vegetation = pd.DataFrame(
        {"county_code": ["001", "001"], "county_name": ["Mombasa", "Mombasa"],
         "year": [2020, 2021], "month": [1, 1], "ndvi": [0.4, 0.5]}
    )
    merged = merge_datasets({"climate": climate, "vegetation": vegetation})
    assert {"precipitation_mm", "ndvi"} <= set(merged.columns)
    assert len(merged) == 2
