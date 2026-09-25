"""Configuration subpackage: typed access to YAML pipeline configuration."""

from .loader import get_pipeline_config, load_pipeline_config

__all__ = ["load_pipeline_config", "get_pipeline_config"]
