"""Dataset schemas / contracts.

Central definition of the county-monthly panel schema so ingestion, validation,
feature engineering and the UI agree on column names and semantics. Keeping
this in one place is the contract that the rest of the pipeline honours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Panel keys (grain = one row per county per month) ---------------------
KEY_COLUMNS: list[str] = ["county_code", "year", "month"]
DATE_COLUMN: str = "date"
COUNTY_NAME_COLUMN: str = "county_name"

# Provenance marker written by the ingestion layer for every observation.
PROVENANCE_COLUMN: str = "data_source"

# Banner surfaced in manifests / the dashboard whenever data is not real.
SYNTHETIC_NOTE: str = (
    "Sample/synthetic data - no real analytical conclusions should be drawn."
)

# --- Raw datasets and their value variables --------------------------------
DATASET_VARIABLES: dict[str, list[str]] = {
    "climate": ["precipitation_mm", "temp_mean_c"],
    "vegetation": ["ndvi", "evi"],
    "agriculture": ["prod_index"],
    "market": ["maize_price_kes_kg"],
    "socioeconomic": ["rural_pop", "poverty_rate", "coping_capacity_index"],
    "outcome": ["food_security_risk_index", "ipc_crisis_households"],
}

# The supervised target for the MVP baseline model.
TARGET: str = "food_security_risk_index"

# Modelled covariates (raw, pre-engineering). Socioeconomic variables are
# county-level and treated as slowly-varying.
BASE_COVARIATES: list[str] = [
    v for k, vs in DATASET_VARIABLES.items()
    if k not in ("outcome",)
    for v in vs
]

ALL_VALUE_COLUMNS: list[str] = [v for vs in DATASET_VARIABLES.values() for v in vs]


@dataclass
class DatasetSpec:
    """Description of one logical dataset in the pipeline."""

    name: str
    file: str
    source: str
    description: str = ""
    variables: list[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        return self.source.lower() == "synthetic"


def dataset_specs(config: dict) -> dict[str, DatasetSpec]:
    """Build :class:`DatasetSpec` objects from the parsed pipeline config."""
    specs: dict[str, DatasetSpec] = {}
    for name, meta in config["datasets"].items():
        specs[name] = DatasetSpec(
            name=name,
            file=meta["file"],
            source=meta["source"],
            description=meta.get("description", ""),
            variables=DATASET_VARIABLES.get(name, []),
        )
    return specs


# --- Food-security risk bands (documented thresholds, NOT model output) ----
# Simple, transparent banding of the 0-100 risk index for UI display. The
# mapping mirrors the IPC phase idea (Minimal -> Crisis -> Above Crisis) but is
# a presentational convenience only; it is not an analytical conclusion.
RISK_BANDS: list[tuple[float, str]] = [
    (25.0, "Watch"),
    (50.0, "Alert"),
    (75.0, "Crisis"),
    (float("inf"), "Above Crisis"),
]


def risk_band(value: float) -> str:
    """Map a 0-100 risk index value to a display band label."""
    for threshold, label in RISK_BANDS:
        if value < threshold:
            return label
    return RISK_BANDS[-1][1]
