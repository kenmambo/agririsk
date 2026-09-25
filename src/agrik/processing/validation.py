"""Schema + quality validation for county-monthly panels.

Validation is *advisory*: it produces a structured :class:`ValidationResult`
separating hard **errors** (structural problems that break downstream code)
from soft **warnings** (out-of-range values, unknown counties, missing
months). Nothing is silently mutated here - the caller decides whether to stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger

LOGGER = get_logger("processing.validation")


@dataclass
class ValidationResult:
    """Outcome of validating one panel DataFrame."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Valid == no structural errors (warnings are tolerated)."""
        return not self.errors

    def summary(self) -> str:
        return f"ValidationResult(errors={len(self.errors)}, warnings={len(self.warnings)})"


def validate_panel(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    value_columns: list[str] | None = None,
) -> ValidationResult:
    """Validate a county-monthly panel against the pipeline contract.

    Parameters
    ----------
    df:
        The panel to validate.
    config:
        Parsed pipeline config (defaults to the cached one).
    value_columns:
        Numeric columns to range-check. Defaults to every column that has a
        rule in ``validation.value_ranges`` and is present in ``df``.
    """
    config = config or get_pipeline_config()
    rules = config.get("validation", {})
    res = ValidationResult()

    keys = list(config.get("panel", {}).get("key_columns", schemas.KEY_COLUMNS))
    required = list(rules.get("required_columns", []))

    # --- Structural: required columns -------------------------------------
    missing = [c for c in required if c not in df.columns]
    if missing:
        res.errors.append(f"Missing required columns: {missing}")
        # Without keys we cannot run the remaining checks meaningfully.
        LOGGER.error("Validation failed: %s", res.errors)
        return res

    # --- Structural: duplicate grain --------------------------------------
    dup = df.duplicated(subset=keys).sum()
    if dup:
        res.errors.append(f"{dup} duplicate rows on grain {keys}")

    # --- county_code membership (warning) ---------------------------------
    known = set(counties.codes())
    unknown = sorted(set(df["county_code"].astype(str).str.zfill(3)) - known)
    if unknown:
        res.warnings.append(f"{len(unknown)} county_code(s) not in registry: {unknown[:5]}")

    # --- year / month ranges ----------------------------------------------
    yr = rules.get("year_range", {})
    if yr and "year" in df.columns:
        bad = df[(df["year"] < yr["min"]) | (df["year"] > yr["max"])]
        if len(bad):
            res.warnings.append(f"{len(bad)} rows have year outside [{yr['min']},{yr['max']}]")
    mo = rules.get("month_range", {})
    if mo and "month" in df.columns:
        bad = df[(df["month"] < mo["min"]) | (df["month"] > mo["max"])]
        if len(bad):
            res.warnings.append(f"{len(bad)} rows have month outside [{mo['min']},{mo['max']}]")

    # --- value ranges + numeric dtype -------------------------------------
    ranges = rules.get("value_ranges", {})
    cols = value_columns or [c for c in ranges if c in df.columns]
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_numeric = int((df[col].notna() & series.isna()).sum())
        if non_numeric:
            res.warnings.append(f"{col}: {non_numeric} non-numeric values coerced to NaN")
        rule = ranges.get(col)
        if rule:
            out = series[(series < rule["min"]) | (series > rule["max"])]
            if len(out):
                res.warnings.append(
                    f"{col}: {len(out)} values outside [{rule['min']},{rule['max']}]"
                )

    # --- completeness of monthly grid (warning) ---------------------------
    if {"year", "month", "county_code"}.issubset(df.columns):
        counts = df.groupby("county_code").size()
        if counts.nunique() > 1:
            res.warnings.append(
                "Counties have unequal month coverage "
                f"(min={int(counts.min())}, max={int(counts.max())}); lags/rolling may NaN."
            )

    for w in res.warnings:
        LOGGER.warning("Validation warning: %s", w)
    for e in res.errors:
        LOGGER.error("Validation error: %s", e)
    LOGGER.info("Validation finished: %s", res.summary())
    return res
