"""AgriRisk Kenya - climate-resilient agriculture early warning platform.

Top-level application package. Layers are deliberately separated:

- :mod:`agrik.settings` / :mod:`agrik.config`  configuration management
- :mod:`agrik.logging`                          structured logging
- :mod:`agrik.counties`                         shared county reference data
- :mod:`agrik.schemas`                          dataset schemas / contracts
- :mod:`agrik.ingestion`                        data ingestion (external boundary)
- :mod:`agrik.processing`                       validation + preprocessing
- :mod:`agrik.features`                         feature engineering pipeline
- :mod:`agrik.models`                           modelling interfaces
- :mod:`agrik.dashboard`                        Streamlit UI (thin consumer)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
