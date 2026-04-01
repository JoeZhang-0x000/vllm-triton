"""Process-wide startup hooks for local vLLM XPU bring-up."""

from __future__ import annotations

from vllm_xpu_host_compat import patch_for_vllm_import

patch_for_vllm_import()
