"""Helpers for bridging host objects and Infinicore tensors."""

from __future__ import annotations

from typing import Any

from vllm_xpu.infinicore import get_infinicore


def to_infinicore_tensor(value: Any, *, dtype: Any | None = None, device: Any | None = None):
    """Convert common host objects to Infinicore tensors when possible."""
    module = get_infinicore()
    if module is None or value is None or hasattr(value, "_underlying"):
        return value

    if hasattr(value, "detach") and hasattr(module, "from_torch"):
        return module.from_torch(value)

    if hasattr(value, "dtype") and hasattr(value, "shape"):
        try:
            return module.from_numpy(value, dtype=dtype, device=device)
        except Exception:
            return value

    if isinstance(value, (list, tuple)):
        try:
            return module.from_list(list(value), dtype=dtype, device=device)
        except Exception:
            return value

    return value
