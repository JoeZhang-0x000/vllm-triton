"""Runtime adapter interfaces and registry helpers."""

from vllm_xpu.runtime.base import BaseRuntimeAdapter, RuntimeAdapter
from vllm_xpu.runtime.registry import (
    NoRuntimeAvailable,
    RuntimeAdapterAlreadyRegistered,
    RuntimeAdapterNotFound,
    RuntimeAdapterUnavailable,
    activate_runtime_adapter,
    clear_runtime_adapters,
    find_available_runtime_adapter,
    get_active_runtime_adapter,
    get_active_runtime_name,
    get_runtime_adapter,
    has_available_runtime_adapter,
    list_runtime_adapters,
    register_runtime_adapter,
    resolve_runtime_adapter,
)

__all__ = [
    "BaseRuntimeAdapter",
    "RuntimeAdapter",
    "NoRuntimeAvailable",
    "RuntimeAdapterAlreadyRegistered",
    "RuntimeAdapterNotFound",
    "RuntimeAdapterUnavailable",
    "activate_runtime_adapter",
    "clear_runtime_adapters",
    "find_available_runtime_adapter",
    "get_active_runtime_adapter",
    "get_active_runtime_name",
    "get_runtime_adapter",
    "has_available_runtime_adapter",
    "list_runtime_adapters",
    "register_runtime_adapter",
    "resolve_runtime_adapter",
]

