"""Data ingestion layer.

Owns everything that touches the outside world for *getting data in*: reading
raw sources, generating clearly-labelled synthetic placeholders, and persisting
raw / processed artefacts. It contains no modelling or UI logic.
"""

from .base import (
    processed_path,
    raw_path,
    read_csv,
    write_csv,
)
from .synthetic import build_synthetic_datasets, write_synthetic_raw

__all__ = [
    "raw_path",
    "processed_path",
    "read_csv",
    "write_csv",
    "build_synthetic_datasets",
    "write_synthetic_raw",
]
