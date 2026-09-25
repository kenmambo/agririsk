"""Streamlit dashboard package.

The UI is a *thin consumer*: it only reads artefacts produced by
:mod:`agrik.pipeline` (feature store + model card) and shared reference data.
It never performs ingestion, validation or model fitting itself.
"""
