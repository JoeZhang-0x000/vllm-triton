"""Minimal XPU worker scaffolding."""

from __future__ import annotations

import os
from typing import Any

from vllm_xpu.infinicore import get_infinicore
from vllm_xpu.runtime.registry import resolve_runtime_adapter
from vllm_xpu.worker.memory import capture_runtime_memory_snapshot
from vllm_xpu.worker.model_runner import XPUModelRunnerBridge

DEFAULT_XPU_WORKER_CLS = "vllm_xpu.worker.worker.XPUWorker"
DEFAULT_XPU_ATTN_BACKEND = "vllm_xpu.attention.backends.flash_attn.XPUAttentionBackend"


class UnsupportedXPUConfiguration(ValueError):
    """Raised when a vLLM config exceeds the supported XPU scope."""


class _NullBackendRunner:
    """Fallback runner used when vLLM is unavailable during local tests."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        del load_dummy_weights
        return None

    def get_model(self) -> Any:
        raise RuntimeError("vLLM is not installed; no backend model runner available")

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        del kv_cache_config
        return None

    def warming_up_model(self) -> None:
        return None

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        del intermediate_tensors
        raise RuntimeError(
            f"vLLM backend runner is not available; cannot execute {scheduler_output!r}"
        )

    def sample_tokens(self, grammar_output: Any) -> Any:
        raise RuntimeError(
            f"vLLM backend runner is not available; cannot sample {grammar_output!r}"
        )


def _maybe_get(obj: Any, attr: str, default: Any = None) -> Any:
    return getattr(obj, attr, default) if obj is not None else default


def _is_bfloat16_dtype(dtype: Any) -> bool:
    if dtype is None:
        return True
    if dtype in ("bfloat16", "bf16"):
        return True
    return str(dtype) in ("torch.bfloat16", "bfloat16")


def validate_xpu_config(vllm_config: Any) -> None:
    """Validate and minimally normalize a config for the XPU foundation."""
    model_config = _maybe_get(vllm_config, "model_config")
    parallel_config = _maybe_get(vllm_config, "parallel_config")
    cache_config = _maybe_get(vllm_config, "cache_config")
    dtype = _maybe_get(model_config, "dtype")

    if not _is_bfloat16_dtype(dtype):
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation only supports bfloat16 dtype"
        )

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
    distributed_executor_backend = _maybe_get(
        parallel_config, "distributed_executor_backend"
    )
    if distributed_executor_backend not in (None, "", "uni"):
        raise UnsupportedXPUConfiguration(
            "vllm_xpu foundation does not support distributed_executor_backend"
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
        self.runtime_device: Any | None = None
        self.device: Any | None = None
        self.host_runner_kind: str | None = None
        self.runtime_name, self.runtime = resolve_runtime_adapter()
        self.model_runner = XPUModelRunnerBridge()

    def init_device(self) -> Any:
        """Initialize the current device through the active runtime adapter."""
        infinicore = get_infinicore()
        if infinicore is not None:
            self.runtime_device = infinicore.device(
                self.runtime.device_type(), self.local_rank
            )
        else:
            self.runtime_device = f"{self.runtime.device_type()}:{self.local_rank}"
        self.runtime.set_device(self.runtime_device)
        self.device = self._make_host_device()
        self.host_runner_kind = self._resolve_host_runner_kind()
        self._init_host_runtime()
        self.model_runner.attach_backend_runner(self._build_backend_runner())
        return self.device

    def get_memory_snapshot(self):
        """Return a minimal runtime memory snapshot."""
        return capture_runtime_memory_snapshot()

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

    def get_model(self) -> Any:
        return self.model_runner.get_model()

    def determine_available_memory(self) -> int:
        cache_config = _maybe_get(self.vllm_config, "cache_config")
        reserved = _maybe_get(cache_config, "kv_cache_memory_bytes")
        if reserved:
            return int(reserved)

        runtime_snapshot = self.get_memory_snapshot()
        gpu_memory_utilization = _maybe_get(cache_config, "gpu_memory_utilization", 0.9)
        if runtime_snapshot.total_memory:
            return int(runtime_snapshot.total_memory * float(gpu_memory_utilization))

        if self.host_runner_kind == "cpu":
            return int(_maybe_get(cache_config, "cpu_kvcache_space_bytes", 0) or 0)

        return 0

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        cache_config = _maybe_get(self.vllm_config, "cache_config")
        if cache_config is not None and hasattr(kv_cache_config, "num_blocks"):
            cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self) -> float:
        return self.model_runner.compile_or_warm_up_model()

    def execute_model(self, scheduler_output: Any) -> Any:
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output: Any) -> Any:
        return self.model_runner.sample_tokens(grammar_output)

    def update_max_model_len(self, max_model_len: int) -> None:
        model_config = _maybe_get(self.vllm_config, "model_config")
        if model_config is not None and hasattr(model_config, "max_model_len"):
            model_config.max_model_len = max_model_len
        self.model_runner.update_max_model_len(max_model_len)

    def take_draft_token_ids(self) -> Any:
        return self.model_runner.take_draft_token_ids()

    def get_cache_block_size_bytes(self) -> int:
        return 0

    def add_lora(self, lora_request: Any) -> bool:
        add_lora = getattr(self.model_runner, "add_lora", None)
        if callable(add_lora):
            return bool(add_lora(lora_request))
        return False

    def remove_lora(self, lora_id: int) -> bool:
        remove_lora = getattr(self.model_runner, "remove_lora", None)
        if callable(remove_lora):
            return bool(remove_lora(lora_id))
        return False

    def pin_lora(self, lora_id: int) -> bool:
        pin_lora = getattr(self.model_runner, "pin_lora", None)
        if callable(pin_lora):
            return bool(pin_lora(lora_id))
        return False

    def list_loras(self) -> set[int]:
        list_loras = getattr(self.model_runner, "list_loras", None)
        if callable(list_loras):
            return set(list_loras())
        return set()

    def _resolve_host_runner_kind(self) -> str:
        override = os.environ.get("VLLM_XPU_HOST_RUNNER", "auto")
        if override in ("cpu", "xpu"):
            return override
        try:
            import torch

            if hasattr(torch, "xpu") and self.runtime.device_type() != "cpu":
                return "xpu"
        except Exception:
            pass
        return "cpu"

    def _make_host_device(self) -> Any:
        try:
            import torch

            if self._resolve_host_runner_kind() == "xpu":
                return torch.device(f"xpu:{self.local_rank}")
            return torch.device("cpu")
        except Exception:
            return f"{self.runtime.device_type()}:{self.local_rank}"

    def _init_host_runtime(self) -> None:
        try:
            from vllm.platforms import current_platform
            from vllm.utils.torch_utils import set_random_seed
            from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )
            model_config = _maybe_get(self.vllm_config, "model_config")
            set_random_seed(_maybe_get(model_config, "seed", 0))
        except Exception:
            return None

    def _build_backend_runner(self) -> Any:
        try:
            if self.host_runner_kind == "xpu":
                from vllm.v1.worker.xpu_model_runner import XPUModelRunner

                return XPUModelRunner(self.vllm_config, self.device)

            from vllm.v1.worker.cpu_model_runner import CPUModelRunner

            return CPUModelRunner(self.vllm_config, self.device)
        except ModuleNotFoundError:
            return _NullBackendRunner()
