"""
config.py
---------
Centralized configuration loading from experiment_config.yaml.
"""

from pathlib import Path
import yaml


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "configs" / "experiment_config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Module-level singleton
CFG = load_config()
