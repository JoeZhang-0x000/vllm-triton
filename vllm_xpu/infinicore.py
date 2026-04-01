"""Helpers for importing the vendored Infinicore Python package."""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType


def _vendored_infinicore_python_path() -> Path:
    return Path(__file__).resolve().parents[1] / "thrid" / "infinicore" / "python"


def ensure_infinicore_on_path() -> None:
    """Add the vendored Infinicore Python package to ``sys.path`` when present."""
    vendor_path = _vendored_infinicore_python_path()
    vendor_path_str = str(vendor_path)
    if vendor_path.is_dir() and vendor_path_str not in sys.path:
        sys.path.insert(0, vendor_path_str)


@lru_cache(maxsize=1)
def get_infinicore() -> ModuleType | None:
    """Return the ``infinicore`` module when importable, else ``None``."""
    ensure_infinicore_on_path()
    try:
        return import_module("infinicore")
    except Exception:
        return None
