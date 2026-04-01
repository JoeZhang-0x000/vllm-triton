"""Runtime-driven memory helpers."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_xpu.runtime.registry import resolve_runtime_adapter


@dataclass(frozen=True)
class RuntimeMemorySnapshot:
    """Small memory snapshot built from the active runtime adapter."""

    runtime_name: str
    device_name: str
    free_memory: int
    total_memory: int


def capture_runtime_memory_snapshot(device_id: int = 0) -> RuntimeMemorySnapshot:
    """Capture a minimal memory snapshot from the resolved runtime adapter."""
    runtime_name, runtime = resolve_runtime_adapter()
    free_memory, total_memory = runtime.mem_get_info()
    return RuntimeMemorySnapshot(
        runtime_name=runtime_name,
        device_name=runtime.get_device_name(device_id),
        free_memory=free_memory,
        total_memory=total_memory,
    )

