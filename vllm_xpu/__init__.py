"""vllm_xpu package bootstrap."""

from vllm_xpu.host_compat import patch_for_vllm_import
from vllm_xpu.bootstrap import initialize

patch_for_vllm_import()
initialize()

__all__ = ["initialize", "patch_for_vllm_import"]
