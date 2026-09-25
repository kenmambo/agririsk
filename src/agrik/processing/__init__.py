"""Processing layer: schema validation + preprocessing (cleaning/merging).

Pure transformations on DataFrames. No IO from external sources (that lives in
``ingestion``) and no modelling. This keeps cleaning logic reusable and unit
testable in isolation.
"""

from .preprocess import clean_dataset, impute_panel, merge_datasets
from .validation import ValidationResult, validate_panel

__all__ = [
    "ValidationResult",
    "validate_panel",
    "clean_dataset",
    "merge_datasets",
    "impute_panel",
]
