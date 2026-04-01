"""Minimal model-runner bridge for the XPU foundation."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_xpu.ops.groups import OperatorGroup
from vllm_xpu.ops.registry import resolve_operator
from vllm_xpu.runtime.registry import resolve_runtime_adapter


@dataclass
class XPUModelRunnerBridge:
    """Small bridge object that resolves runtime and operator providers."""

    runtime_name: str = field(init=False)

    def __post_init__(self) -> None:
        runtime_name, _runtime = resolve_runtime_adapter()
        self.runtime_name = runtime_name

    def resolve_operator_group(self, group: OperatorGroup | str):
        """Resolve an operator implementation for this runner's runtime."""
        return resolve_operator(group, runtime_name=self.runtime_name)

