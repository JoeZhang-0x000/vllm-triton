"""Default Triton operator provider."""

from __future__ import annotations

from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.triton.attention import TritonAttentionOps
from vllm_xpu.ops.triton.linear import TritonLinearOps
from vllm_xpu.ops.triton.norm_act import TritonNormActOps
from vllm_xpu.ops.triton.rotary import TritonRotaryOps

TRITON_PROVIDER_NAME = "triton"


def build_triton_provider() -> OperatorProvider:
    """Build the default Triton operator provider."""
    return OperatorProvider(
        name=TRITON_PROVIDER_NAME,
        implementations={
            "attention": TritonAttentionOps(),
            "linear": TritonLinearOps(),
            "norm_act": TritonNormActOps(),
            "rotary": TritonRotaryOps(),
        },
        metadata={"kind": "default"},
    )

