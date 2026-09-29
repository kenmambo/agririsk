"""Lead-time (forecast horizon) evaluation.

The contemporaneous model answers "what is the risk *this* month?". The
operational early-warning question is "what will the risk be *h months from
now*?". This module re-labels each row's target as the county's risk ``h``
months ahead (a within-county forward shift - features stay leakage-safe
because they only use past information at t) and re-evaluates a freshly fitted
model of the same kind, so horizon degradation is measured honestly:

- the horizon model is trained only on rows whose *feature month* lies in the
  original train block;
- it is tested only on rows whose feature month lies in the original test
  block **and** whose t+h target exists (the last h months per county drop,
  which is why ``n_test`` shrinks with the horizon).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .. import schemas
from ..logging import get_logger
from .evaluation import compute_metrics

LOGGER = get_logger("models.horizon")


def horizon_dataset(
    engineered: pd.DataFrame,
    config: dict[str, Any],
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y_future, dates)`` for a ``horizon``-month lead.

    ``y_future`` at row t is the target at t+horizon *within the same county*;
    rows whose future target is missing are dropped (``make_xy`` NaN policy).
    """
    from ..features import make_xy

    target = config["model"]["target"]
    df = engineered.sort_values(list(config["panel"]["key_columns"])).copy()
    future = df.groupby("county_code", sort=False)[target].shift(-int(horizon))
    df[target] = future
    X, y, _ = make_xy(df, config)
    dates = pd.to_datetime(engineered.loc[X.index, schemas.DATE_COLUMN])
    return X, y, dates


def evaluate_horizon(
    engineered: pd.DataFrame,
    config: dict[str, Any],
    model_factory: Callable[[], Any],
    horizon: int,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> dict[str, float]:
    """Fit/eval a dedicated model for one lead time; returns its metrics.

    ``train_end`` / ``test_start`` are the boundary months of the contemporaneous
    split - the horizon evaluation must not touch rows the final test block
    will use for anything other than testing.
    """
    X, y_future, dates = horizon_dataset(engineered, config, horizon)
    tr = dates <= train_end
    te = dates >= test_start
    if tr.sum() < 10 or te.sum() < 5:
        LOGGER.warning(
            "horizon=%d: too few aligned rows (train=%d, test=%d); skipped.",
            horizon, int(tr.sum()), int(te.sum()),
        )
        return {"horizon": float(horizon), "n_test": float(int(te.sum())), "skipped": 1.0}
    model = model_factory()
    model.fit(X[tr], y_future[tr])
    metrics = compute_metrics(y_future[te].to_numpy(), model.predict(X[te]))
    metrics_out = {
        "horizon": float(horizon),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "n_train": float(int(tr.sum())),
        "n_test": float(int(te.sum())),
        "skipped": 0.0,
    }
    LOGGER.info(
        "horizon=%d months: rmse=%.2f r2=%.3f (n_test=%d).",
        horizon, metrics["rmse"], metrics["r2"], int(te.sum()),
    )
    return metrics_out


def horizon_report(
    engineered: pd.DataFrame,
    config: dict[str, Any],
    model_factory: Callable[[], Any],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
) -> list[dict[str, float]]:
    """Evaluate every configured lead time; returns one row per horizon."""
    horizons = [int(h) for h in config["model"].get("horizons", [1, 3])]
    return [
        evaluate_horizon(
            engineered, config, model_factory, h, train_end, test_start
        )
        for h in horizons
    ]
