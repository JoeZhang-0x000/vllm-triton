"""Minimal model-runner bridge for the XPU foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_xpu.model_executor.conversion import to_infinicore_tensor
from vllm_xpu.model_executor.logits_bridge import export_logits_to_host
from vllm_xpu.ops.groups import OperatorGroup
from vllm_xpu.ops.registry import resolve_operator
from vllm_xpu.runtime.registry import resolve_runtime_adapter


@dataclass
class XPUModelRunnerBridge:
    """Small bridge object that resolves runtime and operator providers."""

    runtime_name: str | None = None
    backend_runner: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.runtime_name is None:
            runtime_name, _runtime = resolve_runtime_adapter()
            self.runtime_name = runtime_name

    def resolve_operator_group(self, group: OperatorGroup | str):
        """Resolve an operator implementation for this runner's runtime."""
        return resolve_operator(group, runtime_name=self.runtime_name)

    def to_device_tensor(
        self,
        value: Any,
        *,
        dtype: Any | None = None,
        device: Any | None = None,
    ) -> Any:
        """Bridge host objects into the active device tensor space when possible."""
        return to_infinicore_tensor(value, dtype=dtype, device=device)

    def logits_to_host(self, logits: Any) -> Any:
        """Export final-step logits back to the host sampling boundary."""
        return export_logits_to_host(logits)

    def attach_backend_runner(self, backend_runner: Any) -> None:
        """Attach the concrete vLLM model runner used for execution."""
        self.backend_runner = backend_runner

    def get_backend_runner(self) -> Any:
        """Return the attached concrete runner or raise a clear bootstrap error."""
        if self.backend_runner is None:
            raise RuntimeError("XPU model runner has not been initialized")
        return self.backend_runner

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.get_backend_runner().load_model(load_dummy_weights=load_dummy_weights)

    def get_model(self) -> Any:
        backend_runner = self.get_backend_runner()
        get_model = getattr(backend_runner, "get_model", None)
        if callable(get_model):
            return get_model()
        return getattr(backend_runner, "model")

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return self.get_backend_runner().get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        backend_runner = self.get_backend_runner()
        initialize_kv_cache = getattr(backend_runner, "initialize_kv_cache", None)
        if callable(initialize_kv_cache):
            initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> float:
        backend_runner = self.get_backend_runner()
        warming_up_model = getattr(backend_runner, "warming_up_model", None)
        if callable(warming_up_model):
            warming_up_model()
        return 0.0

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        return self.get_backend_runner().execute_model(scheduler_output, intermediate_tensors)

    def sample_tokens(self, grammar_output: Any) -> Any:
        return self.get_backend_runner().sample_tokens(grammar_output)

    def update_max_model_len(self, max_model_len: int) -> None:
        backend_runner = self.get_backend_runner()
        update_max_model_len = getattr(backend_runner, "update_max_model_len", None)
        if callable(update_max_model_len):
            update_max_model_len(max_model_len)

    def take_draft_token_ids(self) -> Any:
        backend_runner = self.get_backend_runner()
        take_draft_token_ids = getattr(backend_runner, "take_draft_token_ids", None)
        if callable(take_draft_token_ids):
            return take_draft_token_ids()
        return None

    def __getattr__(self, attr: str) -> Any:
        backend_runner = object.__getattribute__(self, "backend_runner")
        if backend_runner is None:
            raise AttributeError(attr)
        return getattr(backend_runner, attr)
