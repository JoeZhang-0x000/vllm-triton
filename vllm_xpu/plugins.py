"""vLLM plugin entry points for vllm_xpu."""

from __future__ import annotations

from vllm_xpu.bootstrap import initialize
from vllm_xpu.platforms.xpu import XPUPlatform
from vllm_xpu.runtime.registry import has_available_runtime_adapter

XPU_PLATFORM_QUALNAME = "vllm_xpu.platforms.xpu.XPUPlatform"


def register_xpu_platform() -> str | None:
    """Return the XPU platform class when a compatible runtime is available."""
    initialize()
    if not has_available_runtime_adapter():
        return None
    XPUPlatform.sync_runtime_metadata()
    return XPU_PLATFORM_QUALNAME


def register_xpu_general_plugin() -> None:
    """Initialize package-local bootstrap hooks for every vLLM process."""
    initialize()
    if has_available_runtime_adapter():
        XPUPlatform.sync_runtime_metadata()
    return None
