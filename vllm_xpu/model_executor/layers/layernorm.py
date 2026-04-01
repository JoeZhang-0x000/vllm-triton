"""Norm dispatch helpers."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


def apply_rms_norm(
    x: Any,
    weight: Any,
    eps: float = 1e-6,
    *,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch RMSNorm through the operator registry."""
    resolved = resolve_operator("norm_act", runtime_name=runtime_name)
    return resolved.implementation.rms_norm(x, weight, eps=eps)

