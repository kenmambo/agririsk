"""Model-agnostic evaluation utilities.

A temporal split keeps the panel honest: models are trained on earlier months
and tested on later months, mirroring the real forecasting use-case and avoiding
the leakage that a random shuffle would introduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE and R^2 for a regression prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    time_index: pd.Series | np.ndarray,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows into earlier (train) / later (test) by a sortable time index.

    Parameters
    ----------
    time_index:
        Array aligned with ``X``/``y`` used to order rows chronologically
        (e.g. a ``PeriodIndex`` ordinal or ISO date strings).
    test_size:
        Fraction of the chronologically-sorted rows reserved for testing.
    """
    order = np.asarray(time_index).argsort(kind="stable")
    n = len(order)
    n_test = max(1, int(round(n * test_size)))
    test_rows, train_rows = order[-n_test:], order[:-n_test]
    return X.iloc[train_rows], X.iloc[test_rows], y.iloc[train_rows], y.iloc[test_rows]
