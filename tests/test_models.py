"""Tests for the model interface, baseline and evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agrik.models import (
    RidgeRiskModel,
    build_model,
    compute_metrics,
    temporal_train_test_split,
)


def _linear_data():
    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2})
    y = pd.Series(X["a"] * 3 + X["b"] * 1.5 + np.random.default_rng(0).normal(0, 0.5, 20))
    return X, y


def test_ridge_fit_predict_and_score():
    X, y = _linear_data()
    model = RidgeRiskModel(alpha=1.0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    metrics = model.evaluate(X, y)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert np.isfinite(metrics["r2"]) and metrics["r2"] > 0.9


def test_predict_requires_matching_features():
    X, y = _linear_data()
    model = RidgeRiskModel().fit(X, y)
    with pytest.raises(KeyError):
        model.predict(X.drop(columns=["b"]))


def test_temporal_split_is_chronological():
    X, y = _linear_data()
    time_index = pd.date_range("2020-01-01", periods=20, freq="MS").astype(str)
    X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, time_index, test_size=0.25)
    assert len(X_tr) == 15 and len(X_te) == 5
    # The test slice is strictly the most-recent rows.
    assert X_tr.index.max() < X_te.index.min()


def test_compute_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y)
    assert m["rmse"] == 0.0 and m["r2"] == pytest.approx(1.0)


def test_registry_builds_configured_model():
    model = build_model()  # reads model.baseline.type from pipeline.yaml
    assert isinstance(model, RidgeRiskModel)
