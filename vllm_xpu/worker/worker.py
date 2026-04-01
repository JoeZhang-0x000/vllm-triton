"""Minimal XPU worker scaffolding."""

from __future__ import annotations

from typing import Any

from vllm_xpu.runtime.registry import resolve_runtime_adapter
from vllm_xpu.worker.memory import capture_runtime_memory_snapshot
from vllm_xpu.worker.model_runner import XPUModelRunnerBridge

DEFAULT_XPU_WORKER_CLS = "vllm_xpu.worker.worker.XPUWorker"
DEFAULT_XPU_ATTN_BACKEND = "vllm_xpu.attention.backends.flash_attn.XPUAttentionBackend"


class UnsupportedXPUConfiguration(ValueError):
    """Raised when a vLLM config exceeds the supported XPU scope."""


def _maybe_get(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default) if obj is not None else default


def validate_xpu_config(vllm_config: Any) -> None:
    """Validate and minimally normalize a config for the XPU foundation."""
    model_config = _maybe_get(vllm_config, "model_config")
    parallel_config = _maybe_get(vllm_config, "parallel_config")
    cache_config = _maybe_get(vllm_config, "cache_config")

    if _maybe_get(model_config, "quantization") not in (None, "", False):
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation does not support quantization"
        )

    if _maybe_get(vllm_config, "speculative_config") is not None:
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation does not support speculative decoding"
        )

    if _maybe_get(parallel_config, "tensor_parallel_size", 1) > 1:
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation only supports tensor_parallel_size == 1"
        )
    if _maybe_get(parallel_config, "pipeline_parallel_size", 1) > 1:
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation only supports pipeline_parallel_size == 1"
        )
    if _maybe_get(parallel_config, "data_parallel_size", 1) > 1:
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation only supports data_parallel_size == 1"
        )

    if parallel_config is not None and _maybe_get(parallel_config, "worker_cls", "auto") == "auto":
        parallel_config.worker_cls = DEFAULT_XPU_WORKER_CLS

    if cache_config is not None and _maybe_get(cache_config, "block_size") is None:
        cache_config.block_size = 16


class XPUWorker:
    """Small worker shell used by the foundation phase."""

    def __init__(
        self,
        vllm_config: Any,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.device: Any | None = None
        self.runtime_name, self.runtime = resolve_runtime_adapter()
        self.model_runner = XPUModelRunnerBridge()

    def init_device(self) -> Any:
        """Initialize the current device through the active runtime adapter."""
        try:
            import torch

            self.device = torch.device(f"{self.runtime.device_type()}:{self.local_rank}")
        except Exception:  # pragma: no cover - torch-less fallback.
            self.device = f"{self.runtime.device_type()}:{self.local_rank}"
        self.runtime.set_device(self.device)
        return self.device

    def get_memory_snapshot(self):
        """Return a minimal runtime memory snapshot."""
        return capture_runtime_memory_snapshot()

