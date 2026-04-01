"""Minimal attention backend placeholder."""

from __future__ import annotations

from typing import Any

from vllm_xpu.ops.registry import resolve_operator


class XPUAttentionImpl:
    """Placeholder implementation class for the XPU attention backend."""

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        *,
        runtime_name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        resolved = resolve_operator("attention", runtime_name=runtime_name)
        return resolved.implementation.forward(query, key, value, **kwargs)


class XPUAttentionBackend:
    """Minimal attention backend descriptor for foundation wiring."""

    accept_output_buffer = True

    @staticmethod
    def get_name() -> str:
        return "XPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        return XPUAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return []

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_builder_cls():
        return object
