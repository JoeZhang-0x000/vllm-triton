"""Compatibility wrapper around the process-wide host compat module."""

from vllm_xpu_host_compat import patch_for_vllm_import

__all__ = ["patch_for_vllm_import"]
