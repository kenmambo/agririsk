"""Uncertainty and calibration for the risk models (regression flavour).

Split conformal prediction (Vovk et al.) gives *distribution-free* prediction
intervals: fit the model on the train block, take absolute residuals on the
held-out **calibration** block, and widen every future prediction by the
``ceil((n+1)(1-alpha))/n`` empirical quantile of those residuals. On exchangeable
rows the interval covers the true value with probability >= 1-alpha; under a
temporal split that exchangeability is an approximation and is stated as such
wherever intervals are shown.

Also provides the calibration *reliability* diagnostic for regression: bin test
predictions, compare mean predicted vs mean observed per bin, and summarise the
mean gap (lower = better calibrated). No softmax-style "calibration curve"
exists for regression - this is the honest equivalent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def conformal_quantile(residuals: np.ndarray | pd.Series, level: float = 0.9) -> float:
    """Half-width (same +/- units as the target) for a ``level`` interval.

    ``residuals`` must come from data the model was NOT fitted on
    (the calibration block). NaN residuals are dropped.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    a = np.abs(np.asarray(residuals, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("conformal_quantile needs at least one finite residual.")
    q = min(math.ceil(a.size * level) / a.size, 1.0)  # finite-sample correction, clamped
    return float(np.quantile(a, q))


def add_intervals(
    predictions: np.ndarray | pd.Series, halfwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) arrays. Not clipped to the target range: clipping
    would silently narrow the intervals and break the coverage statement."""
    p = np.asarray(predictions, dtype=float)
    return p - halfwidth, p + halfwidth


def interval_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    halfwidth: float,
) -> dict[str, float]:
    """Empirical coverage (PICP) and mean interval width (PINAW) on the test set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    covered = (y_true >= y_pred - halfwidth) & (y_true <= y_pred + halfwidth)
    return {
        "picp": float(np.mean(covered)),
        "pinaw": float(np.mean(2.0 * halfwidth * np.ones_like(y_pred))),
    }


def reliability_table(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Mean predicted vs mean observed risk per prediction-magnitude bin.

    A perfectly calibrated model has ``mean_pred == mean_obs`` on every bin;
    ``calibration_error`` (mean absolute gap over non-empty bins) summarises it
    in target units.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size != y_pred.size or y_true.size == 0:
        raise ValueError("reliability_table needs equal-length non-empty arrays.")
    edges = np.quantile(y_pred, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # include the boundary predictions
    idx = np.clip(np.searchsorted(edges, y_pred, side="left") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "mean_pred": float(np.mean(y_pred[m])),
                "mean_obs": float(np.mean(y_true[m])),
            }
        )
    table = pd.DataFrame(rows)
    err = float(np.mean(np.abs(table["mean_pred"] - table["mean_obs"])))
    table.attrs["calibration_error"] = err
    return table
