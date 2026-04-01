"""Compatibility wrapper for the minimal local generation example."""

from __future__ import annotations

try:
    from examples.basic import main
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from basic import main


if __name__ == "__main__":
    main()
