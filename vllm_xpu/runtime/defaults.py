"""Default runtime registrations for vllm_xpu."""

from __future__ import annotations

from vllm_xpu.infinicore import get_infinicore
from vllm_xpu.runtime.infinicore import InfinicoreRuntimeAdapter
from vllm_xpu.runtime.registry import list_runtime_adapters, register_runtime_adapter


def register_default_runtime_adapters() -> None:
    """Register built-in runtime adapters.

    The Infinicore-first phase ships a single built-in runtime adapter when the
    vendored Infinicore package is importable.
    """
    if "infinicore" in list_runtime_adapters():
        return None
    if get_infinicore() is None:
        return None

    register_runtime_adapter("infinicore", InfinicoreRuntimeAdapter())
    return None
