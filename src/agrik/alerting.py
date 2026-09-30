"""Month-over-month risk escalation - shared by the API and the dashboard.

Early warning in one place: compare a county's mean risk index for an
evaluated month against the mean of a trailing calendar-month baseline
window, and flag counties whose risk rose by ``min_delta`` points OR whose
display band escalated. Both consumers (``agrik.api`` and
``agrik.dashboard``) call :func:`evaluate_alerts` on the already-loaded
feature panel, so neither re-implements - nor drifts from - the rule.

Honesty rules carried here: counties without baseline coverage are counted,
never imputed; zero alerts is a valid answer; band labels are documented
display thresholds, not model output.
"""

from __future__ import annotations

import json

import pandas as pd

from . import schemas

DEFAULT_BASELINE_MONTHS = 3
DEFAULT_MIN_DELTA = 10.0

# Columns of one flagged county, in display order.
ALERT_COLUMNS = [
    "county_code",
    schemas.COUNTY_NAME_COLUMN,
    "risk_current",
    "risk_baseline",
    "delta",
    "band_baseline",
    "band_current",
    "band_escalated",
]

_BAND_ORDER = {label: i for i, (_, label) in enumerate(schemas.RISK_BANDS)}


def month_of(value: object) -> str:
    """ISO month prefix (``YYYY-MM``) of a date-like value."""
    return str(value)[:7]


def latest_month(df: pd.DataFrame) -> str:
    return month_of(df[schemas.DATE_COLUMN].max())


def evaluate_alerts(
    df: pd.DataFrame,
    as_of: str | None = None,
    baseline_months: int = DEFAULT_BASELINE_MONTHS,
    min_delta: float = DEFAULT_MIN_DELTA,
) -> dict:
    """Escalation report for month ``as_of`` (default: latest in ``df``).

    Returns a dict of scalars plus ``alerts`` as a plain list of row dicts.
    Raises ``ValueError`` when the panel has no rows for the evaluated month.
    """
    months = df[schemas.DATE_COLUMN].map(month_of)
    as_of = (as_of or latest_month(df))[:7]
    if not (months == as_of).any():
        raise ValueError(f"no panel rows for month {as_of!r}")
    if baseline_months < 1:
        raise ValueError("baseline_months must be >= 1")

    period = pd.Period(as_of, freq="M")
    start = str(period - int(baseline_months))
    cur = (
        df[months == as_of]
        .groupby(["county_code", schemas.COUNTY_NAME_COLUMN], as_index=False)
        [schemas.TARGET].mean()
        .rename(columns={schemas.TARGET: "risk_current"})
    )
    base = (
        df[(months >= start) & (months < as_of)]
        .groupby("county_code", as_index=False)[schemas.TARGET].mean()
        .rename(columns={schemas.TARGET: "risk_baseline"})
    )
    merged = cur.merge(base, on="county_code", how="left")
    n_no_baseline = int(merged["risk_baseline"].isna().sum())
    merged = merged.dropna(subset=["risk_baseline"])
    merged["delta"] = merged["risk_current"] - merged["risk_baseline"]
    merged["band_current"] = merged["risk_current"].round(1).map(schemas.risk_band)
    merged["band_baseline"] = merged["risk_baseline"].round(1).map(schemas.risk_band)
    merged["band_escalated"] = [
        _BAND_ORDER[c] > _BAND_ORDER[b]
        for c, b in zip(merged["band_current"], merged["band_baseline"], strict=True)
    ]
    flagged = merged[
        (merged["delta"] >= min_delta) | merged["band_escalated"]
    ].copy()
    for col in ("risk_current", "risk_baseline", "delta"):
        flagged[col] = flagged[col].round(2)
    flagged = flagged.sort_values("delta", ascending=False)

    return {
        "as_of": as_of,
        "baseline_window": {
            "from_month": start,
            "to_month": str(period - 1),
            "months": int(baseline_months),
        },
        "min_delta": float(min_delta),
        "rule": (
            f"flagged if risk_current - risk_baseline >= {min_delta} "
            "OR display band escalated vs baseline"
        ),
        "bands_are": "documented display thresholds, not model output",
        "counties_evaluated": int(len(merged)),
        "skipped_no_baseline": n_no_baseline,
        "count": int(len(flagged)),
        # via pandas so numpy scalars become plain JSON-native values.
        "alerts": json.loads(flagged[ALERT_COLUMNS].to_json(orient="records")),
    }
