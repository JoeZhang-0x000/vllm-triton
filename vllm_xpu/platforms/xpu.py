"""Minimal XPU platform placeholder."""

from __future__ import annotations

from typing import Any

from vllm_xpu.runtime.registry import resolve_runtime_adapter
from vllm_xpu.worker.worker import (
    DEFAULT_XPU_ATTN_BACKEND,
    DEFAULT_XPU_WORKER_CLS,
    validate_xpu_config,
)

try:
    from vllm.platforms.interface import Platform, PlatformEnum
except ImportError:  # pragma: no cover - exercised when vllm is absent.
    class Platform:  # type: ignore[no-redef]
        """Fallback stub used when vllm is not installed."""

    class PlatformEnum:  # type: ignore[no-redef]
        """Fallback enum-like stub used when vllm is not installed."""

        OOT = "OOT"


class XPUPlatform(Platform):
    """Minimal out-of-tree XPU platform.

    This class intentionally only provides the narrow delegation and metadata
    needed by the foundation phase. Execution-specific wiring is added later.
    """

    _enum = getattr(PlatformEnum, "OOT", "OOT")
    device_name = "xpu"
    device_type = "xpu"
    dispatch_key = "XPU"
    ray_device_key = "GPU"
    device_control_env_var = "XPU_VISIBLE_DEVICES"
    simple_compile_backend = "eager"
    supported_quantization: list[str] = []
    additional_env_vars: list[str] = []

    @classmethod
    def sync_runtime_metadata(cls) -> None:
        """Project active runtime adapter metadata onto the platform class."""
        _, runtime = resolve_runtime_adapter()
        cls.device_type = runtime.device_type()
        cls.device_name = runtime.device_type()
        cls.dispatch_key = runtime.dispatch_key()

    @classmethod
    def pre_register_and_update(cls, parser: Any = None) -> None:
        try:
            cls.sync_runtime_metadata()
        except Exception:
            return None

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
        validate_xpu_config(vllm_config)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: Any,
        head_size: int,
        dtype: Any,
        kv_cache_dtype: str | None,
        block_size: int,
        use_v1: bool,
        use_mla: bool,
    ) -> str:
        return DEFAULT_XPU_ATTN_BACKEND

    @classmethod
    def set_device(cls, device: Any) -> None:
        _, runtime = resolve_runtime_adapter()
        runtime.set_device(device)

    @classmethod
    def synchronize(cls) -> None:
        _, runtime = resolve_runtime_adapter()
        runtime.synchronize()

    @classmethod
    def empty_cache(cls) -> None:
        _, runtime = resolve_runtime_adapter()
        runtime.empty_cache()

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        _, runtime = resolve_runtime_adapter()
        return runtime.mem_get_info()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        _, runtime = resolve_runtime_adapter()
        return runtime.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return cls.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        _, runtime = resolve_runtime_adapter()
        return runtime.get_device_total_memory(device_id)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> Any | None:
        _, runtime = resolve_runtime_adapter()
        return runtime.get_device_capability(device_id)

    @classmethod
    def inference_mode(cls) -> Any:
        try:
            _, runtime = resolve_runtime_adapter()
        except Exception:
            from contextlib import nullcontext

            return nullcontext()
        return runtime.inference_mode()

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: Any) -> bool:
        return True

    @classmethod
    def get_worker_cls(cls) -> str:
        return DEFAULT_XPU_WORKER_CLS
