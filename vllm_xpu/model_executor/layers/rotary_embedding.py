"""Rotary embedding dispatch helpers."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


def apply_rotary_embedding(
    query: Any,
    key: Any,
    cos: Any,
    sin: Any,
    *,
    position_ids: Any | None = None,
    runtime_name: str | None = None,
) -> Any:
    """Dispatch rotary embedding through the operator registry."""
    resolved = resolve_operator("rotary", runtime_name=runtime_name)
    return resolved.implementation.apply(
        query, key, cos, sin, position_ids=position_ids
    )

