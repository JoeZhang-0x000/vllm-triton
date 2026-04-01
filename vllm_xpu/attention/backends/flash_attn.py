"""Minimal attention backend placeholder."""

from __future__ import annotations


class XPUAttentionImpl:
    """Placeholder implementation class for the XPU attention backend."""


class XPUAttentionBackend:
    """Minimal attention backend descriptor for foundation wiring."""

    accept_output_buffer = True

    @staticmethod
    def get_name() -> str:
        return "XPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        return XPUAttentionImpl

