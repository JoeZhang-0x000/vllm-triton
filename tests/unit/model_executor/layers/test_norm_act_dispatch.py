from __future__ import annotations

from vllm_xpu.model_executor.layers.activation import apply_quick_gelu, apply_silu_mul
from vllm_xpu.model_executor.layers.layernorm import apply_rms_norm
from vllm_xpu.ops.registry import clear_operator_providers, register_default_operator_providers


def setup_function() -> None:
    clear_operator_providers()


def test_norm_act_dispatch_uses_default_triton_provider() -> None:
    register_default_operator_providers()

    rms = apply_rms_norm("x", "weight", eps=1e-6)
    silu = apply_silu_mul("x", "gate")
    qgelu = apply_quick_gelu("x")

    assert rms["provider"] == "infinicore"
    assert rms["op"] == "rms_norm"
    assert silu["provider"] == "infinicore"
    assert silu["op"] == "silu_mul"
    assert qgelu["provider"] == "infinicore"
    assert qgelu["op"] == "quick_gelu"
