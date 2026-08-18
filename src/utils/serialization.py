"""
src/utils/serialization.py
---------------------------
Lightweight serialisation helpers used across the project.

We use the standard-library ``pickle`` module for scikit-learn objects
(e.g. StandardScaler) to avoid an extra dependency.  The module is fully
compatible with joblib-serialised scalers if you later switch to joblib,
because joblib's format is pickle-based.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: str | Path) -> Path:
    """
    Serialise *obj* to a pickle file.

    Parameters
    ----------
    obj  : Any picklable Python object (e.g. a fitted StandardScaler).
    path : Destination file path.

    Returns
    -------
    Resolved Path of the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path.resolve()


def load_pickle(path: str | Path) -> Any:
    """
    Load a pickle file.

    Parameters
    ----------
    path : Path to the pickle file.

    Raises
    ------
    FileNotFoundError : If the file does not exist.

    Returns
    -------
    The deserialised object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path.resolve()}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(path: str | Path, data: dict) -> Path:
    """
    Serialise *data* to a JSON file, converting NumPy scalars / arrays
    and Path objects automatically.

    Parameters
    ----------
    path : Destination file path.
    data : Dictionary to serialise.

    Returns
    -------
    Resolved Path of the saved file.
    """
    import numpy as np

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            if f != f:        # NaN
                return "nan"
            if f == float("inf"):
                return "inf"
            if f == float("-inf"):
                return "-inf"
            return f
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, float):
            if obj != obj:
                return "nan"
            if obj == float("inf") or obj == float("-inf"):
                return str(obj)
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=_convert)
    return path.resolve()


def timestamp_str() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
