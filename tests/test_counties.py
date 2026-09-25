"""Tests for the county reference registry."""

from __future__ import annotations

import agrik.counties as counties


def test_codes_unique_and_zero_padded():
    codes = counties.codes()
    assert len(codes) == len(set(codes))
    assert all(len(c) == 3 and c.isdigit() for c in codes)


def test_frame_columns():
    df = counties.counties_frame()
    assert {"county_code", "name", "lat", "lon", "agro_zone"} <= set(df.columns)
    assert (df["lat"].between(-6, 6)).all()  # Kenya spans roughly the equator
    assert (df["lon"].between(33, 42)).all()


def test_get_county_pads_input():
    assert counties.get_county("1").name == "Mombasa"
    assert counties.get_county("001").code == "001"


def test_get_county_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        counties.get_county("999")
