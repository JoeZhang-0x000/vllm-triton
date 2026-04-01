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
