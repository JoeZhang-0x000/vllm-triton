"""Linear layer dispatch helpers."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


def apply_linear(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch a dense linear operation through the operator registry."""
    resolved = resolve_operator("linear", runtime_name=runtime_name)
    return resolved.implementation.forward(x, weight, bias)

