from __future__ import annotations

from vllm_xpu.model_executor.layers.linear import apply_linear
from vllm_xpu.ops.registry import clear_operator_providers, register_default_operator_providers


def setup_function() -> None:
    clear_operator_providers()


def test_apply_linear_uses_default_triton_provider() -> None:
    register_default_operator_providers()

    output = apply_linear("x", "weight", "bias")

    assert output["provider"] == "triton"
    assert output["x"] == "x"
    assert output["weight"] == "weight"
    assert output["bias"] == "bias"
