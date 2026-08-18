"""
src/utils/paths.py
------------------
Utilities for resolving paths relative to the project root.

The project root is defined as the first ancestor directory of this file
that contains a ``data/`` sub-directory OR a ``configs/`` sub-directory.
This makes all scripts runnable both from the project root and from
``scripts/`` without hard-coding absolute paths.
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Return the absolute path to the project root.

    Strategy: walk upward from ``src/utils/paths.py`` until we find a
    directory that contains ``data/`` or ``configs/``.  Fall back to the
    grandparent of this file if no marker is found.
    """
    here = Path(__file__).resolve().parent  # src/utils/
    for candidate in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        if (candidate / "configs").exists() and (candidate / "scripts").exists() and (candidate / "src").exists():
            return candidate
    # Hard fallback: three levels up from src/utils/paths.py → project root
    return here.parent.parent


def resolve_path(path: str | Path) -> Path:
    """
    Return *path* as an absolute Path.

    If *path* is already absolute, return it unchanged.
    Otherwise, interpret it relative to the project root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (get_project_root() / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """
    Create *path* (and any parents) if it does not exist.

    Returns the resolved Path for convenience.
    """
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
