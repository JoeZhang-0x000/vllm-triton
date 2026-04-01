"""Attention backend definitions for vllm_xpu."""

from vllm_xpu.attention.backends.flash_attn import XPUAttentionBackend

__all__ = ["XPUAttentionBackend"]

