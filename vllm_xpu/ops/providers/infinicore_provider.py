"""Default Infinicore operator provider."""

from __future__ import annotations

from vllm_xpu.ops.infinicore.attention import InfinicoreAttentionOps
from vllm_xpu.ops.infinicore.embedding import InfinicoreEmbeddingOps
from vllm_xpu.ops.infinicore.linear import InfinicoreLinearOps
from vllm_xpu.ops.infinicore.norm_act import InfinicoreNormActOps
from vllm_xpu.ops.infinicore.rotary import InfinicoreRotaryOps
from vllm_xpu.ops.provider import OperatorProvider

INFINICORE_PROVIDER_NAME = "infinicore"


def build_infinicore_provider() -> OperatorProvider:
    """Build the package-default Infinicore operator provider."""
    return OperatorProvider(
        name=INFINICORE_PROVIDER_NAME,
        implementations={
            "attention": InfinicoreAttentionOps(),
            "embedding": InfinicoreEmbeddingOps(),
            "linear": InfinicoreLinearOps(),
            "norm_act": InfinicoreNormActOps(),
            "rotary": InfinicoreRotaryOps(),
        },
        metadata={"kind": "default"},
    )
