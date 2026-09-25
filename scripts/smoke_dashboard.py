"""Headless sanity check for the dashboard data + chart layer.

Verifies the pipeline artefacts load and every Plotly figure builds, without
starting the Streamlit server. Useful in CI: run after `python -m agrik`.
"""

from __future__ import annotations

import json

import pandas as pd

from agrik import counties, schemas
from agrik.dashboard import charts
from agrik.settings import get_settings


def main() -> None:
    s = get_settings()
    df = pd.read_csv(s.features_dir / "features_panel.csv", dtype={"county_code": str})
    df["county_code"] = df["county_code"].str.zfill(3)
    print("features panel:", df.shape)
    assert schemas.TARGET in df.columns

    summary = (
        df.groupby(["county_code", schemas.COUNTY_NAME_COLUMN], as_index=False)[schemas.TARGET]
        .mean()
        .rename(columns={schemas.TARGET: "risk"})
        .merge(counties.counties_frame(), on="county_code", how="left")
    )
    assert charts.county_risk_map(summary) is not None
    assert charts.risk_bars(summary) is not None

    ts = df[df["county_code"] == "001"].sort_values(["year", "month"])
    assert charts.risk_trend(ts) is not None

    card = json.loads((s.models_dir / "model_card.json").read_text(encoding="utf-8"))
    print("model metrics:", card["metrics"])
    print("data note   :", card["data_note"])
    print("SMOKE OK")


if __name__ == "__main__":
    main()
