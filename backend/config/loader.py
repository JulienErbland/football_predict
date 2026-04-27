"""
Config loader for the football prediction pipeline.

Loads settings.yaml, feature_config.yaml, and model_config.yaml from this directory.
Resolves ${ENV_VAR} placeholders by reading from .env (via python-dotenv) and os.environ.
All three configs are exposed as cached singletons so YAML parsing happens only once.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from repo root (two levels up from this file: config/ → backend/ → root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

_CONFIG_DIR = Path(__file__).resolve().parent


def _resolve_env_vars(obj):
    """Recursively replace ${VAR} placeholders with environment variable values."""
    if isinstance(obj, str):
        def replacer(match):
            var = match.group(1)
            value = os.environ.get(var)
            if value is None:
                raise ValueError(
                    f"Config references undefined environment variable: ${{{var}}}. "
                    f"Add it to your .env file or export it in your shell."
                )
            return value
        return re.sub(r"\$\{(\w+)\}", replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def _load_yaml(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    resolved = _resolve_env_vars(raw)
    if "paths" in resolved:
        resolved["paths"] = {k: str(_REPO_ROOT / v) for k, v in resolved["paths"].items()}
    return resolved


@lru_cache(maxsize=1)
def settings() -> dict:
    """Return the main settings config (settings.yaml), resolved and cached."""
    return _load_yaml("settings.yaml")


@lru_cache(maxsize=1)
def feature_config() -> dict:
    """Return the feature config (feature_config.yaml), resolved and cached."""
    return _load_yaml("feature_config.yaml")


@lru_cache(maxsize=1)
def model_config() -> dict:
    """Return the model config (model_config.yaml), resolved and cached."""
    return _load_yaml("model_config.yaml")
