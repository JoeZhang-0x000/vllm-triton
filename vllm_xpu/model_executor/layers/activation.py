"""Activation dispatch helpers."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


def apply_silu_mul(
    x: Any,
    gate: Any,
    *,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch a SiLU-gated activation through the operator registry."""
    resolved = resolve_operator("norm_act", runtime_name=runtime_name)
    return resolved.implementation.silu_mul(x, gate)


def apply_quick_gelu(
    x: Any,
    *,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch QuickGELU through the operator registry."""
    resolved = resolve_operator("norm_act", runtime_name=runtime_name)
    return resolved.implementation.quick_gelu(x)

