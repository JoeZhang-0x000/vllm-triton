from __future__ import annotations

from vllm_xpu.model_executor.layers.rotary_embedding import apply_rotary_embedding
from vllm_xpu.ops.registry import clear_operator_providers, register_default_operator_providers


def setup_function() -> None:
    clear_operator_providers()


def test_rotary_dispatch_uses_default_triton_provider() -> None:
    register_default_operator_providers()

    output = apply_rotary_embedding("q", "k", "cos", "sin", position_ids="p")

    assert output["provider"] == "triton"
    assert output["position_ids"] == "p"

