"""Tests for the validation module."""

from __future__ import annotations

import pandas as pd

from agrik.processing.validation import validate_panel


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_code": ["001", "001"],
            "year": [2020, 2021],
            "month": [1, 2],
            "date": ["2020-01-01", "2021-02-01"],
            "precipitation_mm": [50.0, 80.0],
            "ndvi": [0.4, 0.5],
        }
    )


def test_clean_panel_is_valid():
    assert validate_panel(_panel()).is_valid


def test_missing_required_column_is_error():
    result = validate_panel(_panel().drop(columns=["month"]))
    assert not result.is_valid
    assert any("Missing required" in e for e in result.errors)


def test_duplicate_grain_is_error():
    df = pd.concat([_panel(), _panel().head(1)], ignore_index=True)
    result = validate_panel(df)
    assert any("duplicate" in e.lower() for e in result.errors)


def test_out_of_range_is_warning_not_error():
    df = _panel()
    df.loc[0, "ndvi"] = 5.0  # NDVI must be within [-1, 1]
    result = validate_panel(df)
    assert result.is_valid  # warnings do not invalidate
    assert any("ndvi" in w for w in result.warnings)
