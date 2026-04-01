"""Embedding layer dispatch helpers."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


def apply_embedding(
    indices: Any,
    weight: Any,
    *,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch an embedding lookup through the operator registry."""
    resolved = resolve_operator("embedding", runtime_name=runtime_name)
    return resolved.implementation.forward(indices, weight)
