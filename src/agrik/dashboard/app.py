"""AgriRisk Kenya - Streamlit dashboard (entry point).

Run with:

    streamlit run src/agrik/dashboard/app.py

This is a thin view over artefacts written by ``python -m agrik``. It performs
no ingestion, validation or model training - keeping the UI fully decoupled
from the data-science layers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run .../app.py` even without an editable install.
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agrik import schemas  # noqa: E402
from agrik.dashboard import charts, loaders  # noqa: E402

st.set_page_config(page_title="AgriRisk Kenya", page_icon="🌾", layout="wide")


def _synthetic_banner(df: pd.DataFrame) -> None:
    if loaders.data_is_synthetic(df):
        st.error(
            "⚠️ " + schemas.SYNTHETIC_NOTE.upper()
            + "  Values shown are **synthetic sample data**, not real Kenyan "
            "county observations. Do not use for decisions or conclusions."
        )
    else:
        st.info("Data source: real feeds (see provenance in the Data tab).")


def _county_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["county_code", schemas.COUNTY_NAME_COLUMN], as_index=False)[
            schemas.TARGET
        ]
        .mean()
        .rename(columns={schemas.TARGET: "risk"})
    )
    ref = loaders.county_reference()
    return agg.merge(ref, on="county_code", how="left")


def main() -> None:
    st.title("🌾 AgriRisk Kenya")
    st.caption("County-level food-security early warning — decision-support prototype (MVP).")

    if not loaders.pipeline_ready():
        st.warning("No pipeline artefacts found yet.")
        st.code("pip install -e . && python -m agrik", language="bash")
        st.stop()

    df = loaders.load_feature_panel()
    card = loaders.load_model_card()
    _synthetic_banner(df)

    # ---------------- Sidebar filters ----------------
    st.sidebar.header("Filters")
    years = sorted(pd.to_numeric(df["year"], errors="coerce").dropna().unique().tolist())
    year_lo, year_hi = st.sidebar.slider(
        "Year range", int(min(years)), int(max(years)), (int(min(years)), int(max(years)))
    )
    zones = sorted(df["agro_zone"].dropna().unique().tolist()) if "agro_zone" in df else []
    sel_zones = st.sidebar.multiselect("Agro-ecological zone", zones, default=zones)

    dff = df[
        (pd.to_numeric(df["year"], errors="coerce").between(year_lo, year_hi))
    ]
    if sel_zones and "agro_zone" in dff.columns:
        dff = dff[dff["agro_zone"].isin(sel_zones)]

    summary = _county_summary(dff)

    # ---------------- KPIs ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Counties", dff["county_code"].nunique())
    k2.metric("County-months", f"{len(dff):,}")
    k3.metric("Mean risk index", f"{dff[schemas.TARGET].mean():.1f}")
    k4.metric("Baseline RMSE", f"{card.get('metrics', {}).get('rmse', float('nan')):.2f}")
    k5.metric("Baseline R²", f"{card.get('metrics', {}).get('r2', float('nan')):.2f}")

    tab_map, tab_trend, tab_model, tab_data = st.tabs(
        ["Overview map", "Trends", "Baseline model", "Data & provenance"]
    )

    with tab_map:
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(charts.county_risk_map(summary), use_container_width=True)
        c2.plotly_chart(
            charts.risk_bars(summary, title="Mean risk by county"), use_container_width=True
        )
        st.caption("Map uses approximate county centroids (see county registry).")

    with tab_trend:
        counties_avail = sorted(dff[schemas.COUNTY_NAME_COLUMN].dropna().unique().tolist())
        default = counties_avail[: min(5, len(counties_avail))]
        picked = st.multiselect("Counties to compare", counties_avail, default=default)
        ts = dff[dff[schemas.COUNTY_NAME_COLUMN].isin(picked)].sort_values(
            [schemas.COUNTY_NAME_COLUMN, "year", "month"]
        )
        if not ts.empty:
            st.plotly_chart(charts.risk_trend(ts, title="Food-security risk index over time"),
                            use_container_width=True)
        else:
            st.info("Select at least one county.")

    with tab_model:
        st.subheader("Baseline model card")
        st.json(card)
        st.caption(
            "Metrics describe how well the baseline reproduces the *synthetic* "
            "target and carry no real-world meaning."
        )
        model = loaders.load_baseline_model()
        if model is not None and hasattr(model, "coefficients"):
            st.plotly_chart(charts.driver_coefficients(model.coefficients()),
                            use_container_width=True)

    with tab_data:
        st.subheader("Feature panel (sample)")
        st.dataframe(dff.head(500), use_container_width=True)
        if schemas.PROVENANCE_COLUMN in dff.columns:
            st.write("Provenance values present:", dff[schemas.PROVENANCE_COLUMN].unique().tolist())
        st.download_button(
            "Download filtered panel (CSV)",
            dff.to_csv(index=False).encode(),
            file_name="agrik_filtered_panel.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
