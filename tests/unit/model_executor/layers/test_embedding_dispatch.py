from __future__ import annotations

from vllm_xpu.model_executor.layers.embedding import apply_embedding
from vllm_xpu.ops.registry import clear_operator_providers, register_default_operator_providers


def setup_function() -> None:
    clear_operator_providers()


def test_apply_embedding_uses_default_infinicore_provider() -> None:
    register_default_operator_providers()

    output = apply_embedding("ids", "weight")

    assert output["provider"] == "infinicore"
    assert output["op"] == "embedding"
    assert output["indices"] == "ids"
    assert output["weight"] == "weight"
