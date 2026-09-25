"""Low-level ingestion primitives: paths and CSV read/write.

Concrete source connectors (CHIRPS, MODIS, FEWSNET, KNBS, ...) should be added
as new modules in this package and register themselves in
``config/pipeline.yaml``. Everything shares these IO helpers so provenance and
dtypes stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import schemas
from ..settings import get_settings


def raw_path(filename: str) -> Path:
    """Absolute-ish path to a raw dataset file under ``data/raw``."""
    return get_settings().raw_dir / filename


def processed_path(filename: str) -> Path:
    """Path to a processed dataset file under ``data/processed``."""
    return get_settings().processed_dir / filename


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a panel CSV, enforcing ``county_code`` as a zero-padded string.

    Keeping ``county_code`` as ``str`` avoids silent int-coercion (e.g. ``001``
    becoming ``1``) which would break county joins.
    """
    df = pd.read_csv(path, dtype={schemas.COUNTY_NAME_COLUMN: str})
    # Force county_code to 3-digit zero-padded strings regardless of inference.
    df["county_code"] = df["county_code"].astype(str).str.zfill(3)
    return df


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
