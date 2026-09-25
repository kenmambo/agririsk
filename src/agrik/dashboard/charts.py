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


def driver_coefficients(coefficients: pd.Series, top_n: int = 12) -> go.Figure:
    """Horizontal bar of the baseline model's largest standardised coefficients."""
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
        title=f"Top {top_n} baseline model coefficients (standardised)",
        xaxis_title="Coefficient (effect on risk index, synthetic)",
        margin={"l": 10, "t": 40},
    )
    return fig
