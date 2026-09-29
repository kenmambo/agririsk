"""M3 modelling tests: GBM, three-way split, conformal uncertainty, horizons.

All offline - only sklearn + small synthetic frames (and one full-pipeline
integration run against the labelled synthetic generator in a temp dir).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agrik import schemas
from agrik.models import (
    GradientBoostingRiskModel,
    RidgeRiskModel,
    add_intervals,
    build_model,
    conformal_quantile,
    interval_metrics,
    reliability_table,
    temporal_three_way_split,
)
from agrik.models.horizon import horizon_dataset


def _nonlin_data(n: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    """Saturating (tanh) relationship + noise: easy for trees, capped for Ridge."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.uniform(0.0, 10.0, n), "b": rng.normal(0.0, 1.0, n)})
    y = pd.Series(50.0 + 30.0 * np.tanh(X["a"] - 5.0) + 5.0 * X["b"]
                  + rng.normal(0.0, 1.5, n))
    return X, y


# ---------------------------------------------------------------------------
# gradient boosting model
# ---------------------------------------------------------------------------
def test_gbm_beats_ridge_on_saturating_signal():
    X, y = _nonlin_data()
    X_tr, X_te = X.iloc[:180], X.iloc[180:]
    y_tr, y_te = y.iloc[:180], y.iloc[180:]
    gbm = GradientBoostingRiskModel(max_iter=150).fit(X_tr, y_tr)
    ridge = RidgeRiskModel().fit(X_tr, y_tr)
    gbm_r2 = gbm.evaluate(X_te, y_te)["r2"]
    ridge_r2 = ridge.evaluate(X_te, y_te)["r2"]
    assert gbm_r2 > 0.9
    assert gbm_r2 > ridge_r2


def test_gbm_coefficients_are_permutation_importances():
    X, y = _nonlin_data(n=120)
    gbm = GradientBoostingRiskModel(max_iter=60).fit(X, y)
    imp = gbm.coefficients(n_repeats=3)
    assert len(imp) == X.shape[1]
    # "a" carries the dominant signal, so it must top the importance ranking.
    assert imp.idxmax() == "a"


def test_gbm_handles_nan_rows():
    X, y = _nonlin_data(n=120)
    X = X.astype(float)
    X.iloc[5, 0] = np.nan
    gbm = GradientBoostingRiskModel(max_iter=50).fit(X, y)
    preds = gbm.predict(X)
    assert np.isfinite(preds).all()


def test_registry_builds_gbm_with_own_config_params():
    model = build_model("gbm")
    assert isinstance(model, GradientBoostingRiskModel)
    # hyper-parameters come from the model.gbm YAML section
    assert model.max_iter == 300
    assert model.random_state == 42


def test_registry_rejects_unknown_model():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("xgboost")


def test_model_card_extra_roundtrip():
    X, y = _nonlin_data(n=60)
    gbm = GradientBoostingRiskModel(max_iter=30).fit(X, y)
    card = gbm.model_card({"rmse": 1.0})
    card.extra = {"picp": 0.9, "reliability": [{"bin": 0}]}
    d = card.to_dict()
    assert d["name"] == "gbm"
    assert d["picp"] == 0.9 and d["reliability"] == [{"bin": 0}]


# ---------------------------------------------------------------------------
# three-way chronological split
# ---------------------------------------------------------------------------
def _split_data(n: int = 40):
    X = pd.DataFrame({"a": np.arange(float(n))})
    y = pd.Series(np.arange(float(n)))
    dates = pd.date_range("2020-01-01", periods=n, freq="MS").astype(str)
    return X, y, np.asarray(dates)


def test_three_way_split_is_chronological_and_disjoint():
    X, y, t = _split_data(40)
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = temporal_three_way_split(
        X, y, t, test_size=0.25, cal_size=0.15
    )
    assert len(X_tr) == 24 and len(X_cal) == 6 and len(X_te) == 10
    assert X_tr["a"].max() < X_cal["a"].min()
    assert X_cal["a"].max() < X_te["a"].min()
    assert len(set(X_tr.index) & set(X_te.index)) == 0


def test_three_way_split_rejects_degenerate_sizes():
    X, y, t = _split_data(10)
    with pytest.raises(ValueError, match="no training rows"):
        temporal_three_way_split(X, y, t, test_size=0.5, cal_size=0.5)


# ---------------------------------------------------------------------------
# conformal uncertainty + reliability
# ---------------------------------------------------------------------------
def test_conformal_quantile_coverage_on_holdout():
    rng = np.random.default_rng(1)
    cal = rng.normal(0.0, 3.0, 200)      # model NOT fitted on these residuals
    q = conformal_quantile(cal, level=0.9)
    hold = rng.normal(0.0, 3.0, 500)
    coverage = float(np.mean(np.abs(hold) <= q))
    assert coverage >= 0.85  # nominal 90% -> empirical close, never far below


def test_conformal_quantile_validation():
    with pytest.raises(ValueError, match="level"):
        conformal_quantile(np.array([1.0]), level=1.5)
    with pytest.raises(ValueError, match="at least one"):
        conformal_quantile(np.array([np.nan, np.inf]))


def test_add_intervals_and_interval_metrics():
    preds = np.array([50.0, 60.0, 70.0])
    lo, hi = add_intervals(preds, 5.0)
    assert lo.tolist() == [45.0, 55.0, 65.0]
    assert hi.tolist() == [55.0, 65.0, 75.0]
    y = np.array([52.0, 64.0, 90.0])  # first two inside +/-5, third misses
    m = interval_metrics(y, preds, 5.0)
    assert m["picp"] == pytest.approx(2 / 3)
    assert m["pinaw"] == pytest.approx(10.0)


def test_reliability_table_perfect_and_biased():
    y = np.array([10.0, 30.0, 50.0, 70.0, 90.0] * 10)
    rel = reliability_table(y, y.copy(), n_bins=5)
    assert rel.attrs["calibration_error"] == pytest.approx(0.0)
    assert rel["n"].sum() == len(y)
    biased = reliability_table(y, y + 10.0, n_bins=5)
    assert biased.attrs["calibration_error"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# horizon dataset (within-county forward shift)
# ---------------------------------------------------------------------------
def _tiny_engineered() -> pd.DataFrame:
    rows = []
    for county, base in (("001", 0.0), ("002", 100.0)):
        for m in range(1, 7):
            rows.append({
                "county_code": county, "year": 2020, "month": m,
                "date": f"2020-{m:02d}-01", "county_name": f"C{county}",
                schemas.PROVENANCE_COLUMN: "synthetic",
                "f": base + m,                       # feature
                schemas.TARGET: base + 10 * m,       # target
            })
    return pd.DataFrame(rows)


def test_horizon_dataset_shifts_within_counties():
    df = _tiny_engineered()
    cfg = {
        "model": {"target": schemas.TARGET},
        "panel": {"key_columns": ["county_code", "year", "month"]},
    }
    X, y_future, dates = horizon_dataset(df, cfg, horizon=1)
    assert len(X) == 10  # last month per county drops (no t+1 target)
    assert list(X.columns) == ["f"]
    cc = df.loc[X.index, "county_code"]
    # county 001 row at month m must carry ITS OWN month m+1 target.
    c1 = cc == "001"
    expected = [10.0 * (i + 2) for i in range(5)]  # m=2..6 targets for t=1..5
    assert y_future[c1].tolist() == expected
    # no value from county 001 ever bled into county 002's horizon labels
    c2 = cc == "002"
    assert all(v >= 110.0 for v in y_future[c2])
    assert dates.index.equals(X.index)


def test_horizon_longer_lead_drops_more_rows():
    df = _tiny_engineered()
    cfg = {
        "model": {"target": schemas.TARGET},
        "panel": {"key_columns": ["county_code", "year", "month"]},
    }
    X1, _, _ = horizon_dataset(df, cfg, 1)
    X3, _, _ = horizon_dataset(df, cfg, 3)
    assert len(X3) == 6 < len(X1)


# ---------------------------------------------------------------------------
# pipeline integration (synthetic, temp dir): all M3 artefacts exist
# ---------------------------------------------------------------------------
def test_step_train_writes_m3_artefacts(isolated_env):
    from agrik.pipeline import run_pipeline

    summary = run_pipeline(force_raw=True, seed=3)
    from agrik.settings import get_settings

    s = get_settings()
    metrics = summary["metrics"]
    assert metrics["n_cal"] > 0
    assert 0.0 <= metrics["picp"] <= 1.0
    assert metrics["pinaw"] > 0.0

    comparison = json.loads((s.models_dir / "comparison.json").read_text(encoding="utf-8"))
    names = {r["name"] for r in comparison["rows"]}
    assert {"ridge", "gbm"} <= names
    assert (s.models_dir / "baseline_ridge.joblib").exists()
    assert (s.models_dir / "baseline_gbm.joblib").exists()

    preds = pd.read_csv(s.models_dir / "test_predictions.csv")
    assert {"y_true", "y_pred", "y_lo", "y_hi"} <= set(preds.columns)
    assert (preds["y_lo"] <= preds["y_pred"] + 1e-9).all()
    assert (preds["y_pred"] <= preds["y_hi"] + 1e-9).all()

    card = json.loads((s.models_dir / "model_card.json").read_text(encoding="utf-8"))
    assert 0 < card["interval_level"] < 1
    assert len(card["reliability"]) >= 2
    assert card["calibration_error"] >= 0.0
    rows = card["horizon_metrics"]
    assert [r["horizon"] for r in rows] == [1.0, 3.0]
    # honest: a longer lead has fewer aligned test rows, never more
    assert rows[1]["n_test"] <= rows[0]["n_test"]
    # the model card still carries the loud provenance note
    assert "synthetic" in card["data_note"].lower()
