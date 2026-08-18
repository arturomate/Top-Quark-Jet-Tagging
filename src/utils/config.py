"""
src/utils/config.py
-------------------
Helpers for loading YAML experiment configs and serialising results to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """
    Load a YAML experiment config using ``yaml.safe_load``.

    Parameters
    ----------
    path : Path to the YAML file (str or Path).

    Returns
    -------
    cfg : Nested dict mirroring the YAML structure.

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def save_json(path: str | Path, data: dict) -> None:
    """
    Serialise *data* to a JSON file, converting NumPy / Python scalars
    to JSON-safe types automatically.

    Parameters
    ----------
    path : Destination file path.
    data : Dictionary to serialise.
    """
    import numpy as np

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return str(obj)          # NaN / ±Inf → string
        if isinstance(obj, Path):
            return str(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_convert)
