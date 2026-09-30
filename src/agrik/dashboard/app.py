"""AgriRisk Kenya - Streamlit dashboard (entry point).

Run with:

    streamlit run src/agrik/dashboard/app.py

This is a thin view over artefacts written by ``python -m agrik``. It performs
no ingestion, validation or model training - keeping the UI fully decoupled
from the data-science layers. The early-warning rule itself lives in the
shared ``agrik.alerting`` module (same code path the API ``/alerts`` route
uses), so this view only renders - never re-implements - it.
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

from agrik import alerting, schemas  # noqa: E402
from agrik.dashboard import charts, loaders  # noqa: E402

st.set_page_config(page_title="AgriRisk Kenya", page_icon="🌾", layout="wide")


def _synthetic_banner(df: pd.DataFrame) -> None:
    sources = loaders.dataset_sources(df)
    real = {n: s for n, s in sources.items() if "synthetic" not in s}
    synth = {n: s for n, s in sources.items() if "synthetic" in s}
    if loaders.data_is_synthetic(df) and real:
        st.warning(
            "⚠️ **Partially real data.** Real feeds: "
            + ", ".join(f"**{n}** ({s})" for n, s in sorted(real.items()))
            + ". Still SYNTHETIC: "
            + ", ".join(f"**{n}**" for n in sorted(synth))
            + " - do not draw analytical conclusions from synthetic components."
        )
    elif loaders.data_is_synthetic(df):
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

    tab_alerts, tab_map, tab_trend, tab_model, tab_data = st.tabs(
        ["⚠ Early warning", "Overview map", "Trends", "Baseline model",
         "Data & provenance"]
    )

    with tab_alerts:
        st.subheader("Month-over-month risk escalation")
        months = sorted(df[schemas.DATE_COLUMN].astype(str).str.slice(0, 7).unique())
        as_of = st.selectbox(
            "Evaluate month (YYYY-MM)", months, index=len(months) - 1,
        )
        b1, b2 = st.columns(2)
        baseline_months = b1.slider("Baseline window (trailing months)", 1, 12, 3)
        min_delta = b2.slider("Alert threshold (risk points)", 0.0, 40.0, 10.0, step=0.5)
        try:
            report = alerting.evaluate_alerts(
                df, as_of=as_of, baseline_months=baseline_months,
                min_delta=min_delta,
            )
        except ValueError as exc:
            st.warning(f"Cannot evaluate {as_of}: {exc}")
            report = None

        if report is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alerts raised", report["count"])
            m2.metric("Counties evaluated", report["counties_evaluated"])
            m3.metric("Skipped (no baseline)", report["skipped_no_baseline"])
            m4.metric("Baseline window", report["baseline_window"]["from_month"]
                      + " → " + report["baseline_window"]["to_month"])

            alerts_df = pd.DataFrame(report["alerts"])
            if alerts_df.empty:
                st.success(
                    f"No county escalated vs its {baseline_months}-month baseline "
                    f"in {as_of} (rule: Δ ≥ {min_delta} or band jump). Zero alerts "
                    "is a valid early-warning answer, not missing data."
                )
            else:
                st.dataframe(
                    alerts_df[[
                        "county_code", schemas.COUNTY_NAME_COLUMN, "risk_baseline",
                        "risk_current", "delta", "band_baseline", "band_current",
                        "band_escalated",
                    ]],
                    use_container_width=True, hide_index=True,
                )
                ref = loaders.county_reference()
                mapped = alerts_df.merge(ref[["county_code", "lat", "lon", "agro_zone"]],
                                         on="county_code", how="left")
                mapped = mapped.dropna(subset=["lat", "lon"])
                c1, c2 = st.columns([1, 1])
                c1.plotly_chart(charts.alert_escalation(
                    mapped, title=f"Escalating counties — {as_of}"),
                    use_container_width=True)
                if not mapped.empty:
                    c2.plotly_chart(charts.alert_map(
                        mapped, title="Where the alerts are"), use_container_width=True)
            st.caption(
                f"Rule: {report['rule']}. {report['bands_are']}. "
                "Counties without baseline coverage are counted, never imputed."
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
        st.subheader(f"Primary model: {card.get('name', 'ridge')}")
        m = card.get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test RMSE", f"{m.get('rmse', float('nan')):.2f}")
        c2.metric("Test R²", f"{m.get('r2', float('nan')):.3f}")
        c3.metric(f"Conformal coverage (target {int(card.get('interval_level', 0.9) * 100)}%)",
                  f"{m.get('picp', float('nan')):.2f}")
        c4.metric("Calibration error",
                  f"{card.get('calibration_error', float('nan')):.1f}")
        st.caption(card.get(
            "data_note",
            "Metrics describe how well the baseline reproduces the target.",
        ))

        comparison = loaders.load_comparison()
        if comparison.get("rows"):
            st.subheader("Model comparison - identical chronological split")
            st.dataframe(pd.DataFrame(comparison["rows"]), use_container_width=True)

        preds = loaders.load_test_predictions()
        if not preds.empty:
            agg = (
                preds.groupby("date", as_index=False)[
                    ["y_true", "y_pred", "y_lo", "y_hi"]].mean()
            )
            level_pct = int(card.get("interval_level", 0.9) * 100)
            st.plotly_chart(
                charts.prediction_band(agg, level_pct=level_pct),
                use_container_width=True,
            )
            st.caption(
                "Held-out (test-block) months, averaged across counties. "
                + card.get("interval_note", "")
            )
        if card.get("reliability"):
            st.plotly_chart(
                charts.reliability_chart(card["reliability"],
                                         card.get("calibration_error")),
                use_container_width=True,
            )
        if card.get("horizon_metrics"):
            st.plotly_chart(
                charts.horizon_degradation(card["horizon_metrics"]),
                use_container_width=True,
            )
            st.caption(
                "Each lead time re-fits the model on the train block only and "
                "tests on later months whose t+h target exists (n_test shrinks "
                "with the horizon - honest, not extrapolated)."
            )

        model = loaders.load_baseline_model()
        if model is not None and hasattr(model, "coefficients"):
            imp_title = (
                "Top drivers - permutation importances (predictive, not causal)"
                if getattr(model, "name", "") == "gbm" else ""
            )
            st.plotly_chart(charts.driver_coefficients(model.coefficients(),
                                                       title=imp_title),
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
