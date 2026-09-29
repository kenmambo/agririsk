"""Plotly figure builders for the dashboard (pure view layer).

Each function takes already-prepared DataFrames/Series and returns a
``plotly.graph_objects`` figure. No data loading or transformation happens here.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def county_risk_map(summary: pd.DataFrame, title: str = "") -> go.Figure:
    """Bubble map of average risk by county (approximate centroids)."""
    fig = px.scatter_map(
        summary,
        lat="lat",
        lon="lon",
        color="risk",
        size="risk",
        hover_name="county_name",
        hover_data=["agro_zone", "risk"],
        color_continuous_scale="YlOrRd",
        map_style="open-street-map",
        zoom=5,
        center={"lat": summary["lat"].mean(), "lon": summary["lon"].mean()},
        title=title,
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return fig


def risk_trend(timeseries: pd.DataFrame, title: str = "") -> go.Figure:
    """Line chart of the risk index over time for one or more counties."""
    fig = px.line(
        timeseries,
        x="date",
        y="food_security_risk_index",
        color="county_name",
        markers=True,
        title=title,
    )
    fig.update_layout(yaxis_title="Risk index (0-100)", xaxis_title="", margin={"l": 40, "t": 40})
    return fig


def risk_bars(summary: pd.DataFrame, title: str = "") -> go.Figure:
    """Ranked bar chart of average risk across counties."""
    ordered = summary.sort_values("risk", ascending=False)
    fig = px.bar(
        ordered,
        x="risk",
        y="county_name",
        orientation="h",
        color="risk",
        color_continuous_scale="YlOrRd",
        title=title,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), margin={"l": 10, "t": 40})
    return fig


def driver_coefficients(coefficients: pd.Series, top_n: int = 12,
                        title: str = "") -> go.Figure:
    """Horizontal bar of the primary model's largest drivers.

    For Ridge these are standardised coefficients; for tree models the
    pipeline passes permutation importances (pass an explicit ``title``).
    """
    top = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)[:top_n]
    fig = go.Figure(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color=["#c0392b" if v < 0 else "#27ae60" for v in top.values],
        )
    )
    fig.update_layout(
        title=title or f"Top {top_n} baseline model coefficients (standardised)",
        xaxis_title="Value on risk index - predictive, not causal",
        margin={"l": 10, "t": 40},
    )
    return fig


def prediction_band(agg: pd.DataFrame, level_pct: int = 90,
                    title: str = "Held-out predictions vs observed") -> go.Figure:
    """Observed vs predicted risk over the test block with a conformal band.

    ``agg`` has columns date, y_true, y_pred, y_lo, y_hi (already aggregated
    per month across counties - county-level bands would be unreadable)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["y_hi"], mode="lines",
        line=dict(color="rgba(255,165,0,0.25)", width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["y_lo"], mode="lines",
        line=dict(color="rgba(255,165,0,0.25)", width=0),
        fill="tonexty", name=f"{level_pct}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["y_true"], mode="lines+markers", name="observed",
        line=dict(color="#2c3e50"),
    ))
    fig.add_trace(go.Scatter(
        x=agg["date"], y=agg["y_pred"], mode="lines", name="predicted",
        line=dict(color="#e67e22", dash="dot"),
    ))
    fig.update_layout(
        title=title, yaxis_title="Mean risk index (0-100)", xaxis_title="",
        margin={"l": 40, "t": 40},
    )
    return fig


def reliability_chart(bins: list[dict], cal_err: float | None = None) -> go.Figure:
    """Mean predicted vs mean observed risk per prediction bin (regression
    calibration diagnostic - equal bars per bin means well calibrated)."""
    df = pd.DataFrame(bins)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["mean_pred"], y=df["bin"], orientation="h",
                         name="mean predicted", marker_color="#3498db"))
    fig.add_trace(go.Bar(x=df["mean_obs"], y=df["bin"], orientation="h",
                         name="mean observed", marker_color="#e74c3c"))
    fig.update_layout(
        barmode="group",
        title=("Calibration reliability (predicted vs observed by bin)"
               + (f" - mean gap {cal_err:.1f}" if cal_err is not None else "")),
        xaxis_title="Mean risk index", yaxis_title="Prediction bin (low -> high)",
        margin={"l": 10, "t": 40},
    )
    return fig


def horizon_degradation(rows: list[dict], metric: str = "rmse") -> go.Figure:
    """Test metric vs forecast lead time (months ahead) - honesty chart:
    early-warning value decays with the horizon."""
    df = pd.DataFrame([r for r in rows if not r.get("skipped")])
    fig = go.Figure(go.Scatter(
        x=df["horizon"], y=df[metric], mode="lines+markers",
        line=dict(color="#8e44ad"), marker=dict(size=10),
    ))
    fig.update_layout(
        title=f"Forecast quality by lead time ({metric})",
        xaxis_title="Lead time (months ahead)", yaxis_title=metric.upper(),
        margin={"l": 40, "t": 40},
    )
    return fig
