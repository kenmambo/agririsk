"""Preprocessing primitives: cleaning, merging, imputation.

These are deterministic, side-effect-free DataFrame transforms. They deliberately
do NOT compute derived model features (that is the ``features`` layer's job) -
this separation keeps leakage control and testing tractable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .. import schemas
from ..config import get_pipeline_config
from ..logging import get_logger

LOGGER = get_logger("processing.preprocess")


def clean_dataset(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    value_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Coerce dtypes, drop exact grain duplicates and sort the panel.

    Parameters
    ----------
    df:
        A single raw dataset (long county-monthly format).
    value_columns:
        Columns to coerce to numeric. Defaults to any configured value column
        present in ``df``.
    """
    config = config or get_pipeline_config()
    keys = list(config.get("panel", {}).get("key_columns", schemas.KEY_COLUMNS))
    df = df.copy()

    # county_code as zero-padded string; keys as integers.
    df["county_code"] = df["county_code"].astype(str).str.zfill(3)
    for k in ("year", "month"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")

    cols = value_columns or [
        c for c in schemas.ALL_VALUE_COLUMNS if c in df.columns
    ]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset=keys, keep="first")
    dropped = before - len(df)
    if dropped:
        LOGGER.warning("clean_dataset dropped %d duplicate grain rows.", dropped)

    date_col = config.get("panel", {}).get("date_column", schemas.DATE_COLUMN)
    sort_by = keys if date_col not in df.columns else keys
    df = df.sort_values(sort_by).reset_index(drop=True)
    return df


def merge_datasets(
    datasets: dict[str, pd.DataFrame],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Merge per-dataset frames into one master panel on the shared grain.

    All datasets must already be cleaned with :func:`clean_dataset`. Context
    columns (``county_name``, ``date``) are carried once from the first frame
    that provides them. Each dataset contributes its *value* columns **and**
    its provenance as a ``{dataset}_source`` column, so per-dataset real vs
    synthetic status survives the merge. The overall ``data_source`` column is
    the single source name, or ``"mixed"`` when datasets differ. Uses an outer
    join so partial coverage across sources is preserved (and visible).
    """
    config = config or get_pipeline_config()
    keys = list(config.get("panel", {}).get("key_columns", schemas.KEY_COLUMNS))
    if not datasets:
        raise ValueError("merge_datasets received no datasets.")

    date_col = config.get("panel", {}).get("date_column", schemas.DATE_COLUMN)
    context_cols = [schemas.COUNTY_NAME_COLUMN, date_col]

    frames = list(datasets.values())
    base_cols = keys + [c for c in context_cols if c in frames[0].columns]
    result = frames[0][base_cols].copy()

    sources_seen: set[str] = set()
    for name, d in datasets.items():
        value_cols = [
            c for c in d.columns
            if c not in keys and c not in context_cols and c != schemas.PROVENANCE_COLUMN
        ]
        merge_cols = keys + value_cols
        if schemas.PROVENANCE_COLUMN in d.columns:
            merge_cols = merge_cols + [schemas.PROVENANCE_COLUMN]
        subset = d[merge_cols].rename(
            columns={schemas.PROVENANCE_COLUMN: f"{name}_source"}
        )
        if schemas.PROVENANCE_COLUMN in d.columns:
            sources_seen.update(d[schemas.PROVENANCE_COLUMN].astype(str).unique())
        if len(value_cols) == 0 and f"{name}_source" not in subset.columns:
            continue
        result = result.merge(subset, on=keys, how="outer")

    result[schemas.PROVENANCE_COLUMN] = (
        next(iter(sources_seen)) if len(sources_seen) == 1 else "mixed"
    )
    if schemas.COUNTY_NAME_COLUMN not in result.columns:
        raise ValueError("Merged panel is missing county_name.")
    LOGGER.info(
        "Merged %d datasets (sources: %s) -> master panel %s rows.",
        len(frames), sorted(sources_seen) or ["n/a"], len(result),
    )
    return result.sort_values(keys).reset_index(drop=True)


def impute_panel(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Linearly interpolate gaps *within each county* over the monthly series.

    Interpolation is per-county and only fills interior gaps (``limit_area``
    behaviour via sorting on year/month). Boundary NaNs are then forward/
    backward filled. This is a light-touch cleaning step; genuinely missing
    source data should be handled at ingestion.
    """
    config = config or get_pipeline_config()
    keys = list(config.get("panel", {}).get("key_columns", schemas.KEY_COLUMNS))
    cols = columns or [c for c in schemas.ALL_VALUE_COLUMNS if c in df.columns]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values(keys).reset_index(drop=True).copy()

    # Interpolate within each county over its (already time-sorted) rows,
    # filling interior gaps then boundaries. Direct per-group assignment keeps
    # the county_code column intact (groupby.apply would drop it as the key).
    if cols:
        for group_idx in df.groupby("county_code").groups.values():
            idx = list(group_idx)
            df.loc[idx, cols] = (
                df.loc[idx, cols].interpolate(method="linear").ffill().bfill().to_numpy()
            )

    remaining = int(df[cols].isna().sum().sum()) if cols else 0
    if remaining:
        LOGGER.warning("impute_panel left %d NaN cells (empty counties?).", remaining)
    return df
