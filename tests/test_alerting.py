"""Unit tests for the shared alerting rule (``agrik.alerting``).

Crafted mini-panels (offline, no pipeline run) so every branch of the rule
is pinned down exactly: delta alerts, band escalations, honest skipping of
counties without baseline coverage, and error paths. The API and the
dashboard both consume this module, so testing it here covers both.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agrik import alerting, schemas


def make_panel(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """rows = (county_code, county_name, date, risk) -> panel-shaped frame."""
    df = pd.DataFrame(
        rows,
        columns=["county_code", schemas.COUNTY_NAME_COLUMN,
                 schemas.DATE_COLUMN, schemas.TARGET],
    )
    return df


def three_month_panel() -> pd.DataFrame:
    # c1: 20 -> 60 at 2022-02 (delta 40, big band jump) -> delta alert.
    # c2: 24 -> 27 (delta 3) but Watch -> Alert band edge -> escalation alert.
    # c3: stable 10 throughout -> never flagged.
    # c4: appears only in the evaluated month -> no baseline, skipped honestly.
    rows = []
    for month, (v1, v2, v3) in {
        "2021-11": (20.0, 24.0, 10.0),
        "2021-12": (20.0, 24.0, 10.0),
        "2022-01": (20.0, 24.0, 10.0),
        "2022-02": (60.0, 27.0, 10.0),
    }.items():
        rows += [
            ("001", "County One", month, v1),
            ("002", "County Two", month, v2),
            ("003", "County Three", month, v3),
        ]
    rows.append(("004", "County Four", "2022-02", 90.0))
    return make_panel(rows)


def test_evaluate_alerts_flags_delta_and_escalation():
    report = alerting.evaluate_alerts(three_month_panel(), as_of="2022-02")
    assert report["as_of"] == "2022-02"
    assert report["baseline_window"] == {
        "from_month": "2021-11", "to_month": "2022-01", "months": 3,
    }
    # c1 (delta 40) and c2 (band escalation) flagged; c3 stable is not;
    # c4 has no baseline so it is skipped, counted honestly.
    codes = [a["county_code"] for a in report["alerts"]]
    assert codes == ["001", "002"]  # sorted by delta, descending
    assert report["count"] == len(report["alerts"]) == 2
    assert report["counties_evaluated"] == 3
    assert report["skipped_no_baseline"] == 1
    first = report["alerts"][0]
    assert first["risk_current"] == 60.0
    assert first["risk_baseline"] == 20.0
    assert first["delta"] == 40.0
    assert first["band_escalated"] is True
    assert set(report["alerts"][0]) == set(alerting.ALERT_COLUMNS)


def test_evaluate_alerts_zero_is_valid_and_json_native():
    # Quiet panel: everything flat -> empty alerts, still a clean report.
    flat = make_panel([
        ("001", "County One", "2022-01", 30.0),
        ("001", "County One", "2022-02", 30.0),
        ("002", "County Two", "2022-01", 31.0),
        ("002", "County Two", "2022-02", 30.5),
    ])
    report = alerting.evaluate_alerts(flat, as_of="2022-02", baseline_months=1)
    assert report["count"] == 0
    assert report["alerts"] == []
    import json
    json.dumps(report)  # numpy-free, directly serializable


def test_evaluate_alerts_min_delta_suppresses_small_deltas():
    report = alerting.evaluate_alerts(
        three_month_panel(), as_of="2022-02", min_delta=100.0
    )
    # Only pure band escalations survive a delta threshold nothing can meet.
    codes = [a["county_code"] for a in report["alerts"]]
    assert "001" in codes  # still escalated (band rule ORs with delta)
    assert all(a["delta"] >= 100.0 or a["band_escalated"]
               for a in report["alerts"])


def test_evaluate_alerts_first_window_honest_no_baseline():
    df = make_panel([
        ("001", "County One", "2022-01", 55.0),
        ("002", "County Two", "2022-01", 80.0),
    ])
    report = alerting.evaluate_alerts(df)  # default as_of = latest = 2022-01
    assert report["counties_evaluated"] == 0
    assert report["skipped_no_baseline"] == 2
    assert report["alerts"] == []


def test_evaluate_alerts_errors():
    df = three_month_panel()
    with pytest.raises(ValueError, match="no panel rows"):
        alerting.evaluate_alerts(df, as_of="2030-01")
    with pytest.raises(ValueError, match="baseline_months"):
        alerting.evaluate_alerts(df, as_of="2022-02", baseline_months=0)


def test_latest_month_and_month_of():
    df = three_month_panel()
    assert alerting.latest_month(df) == "2022-02"
    assert alerting.month_of("2022-02-15") == "2022-02"
    assert alerting.month_of(pd.Timestamp("2022-02-15")) == "2022-02"
