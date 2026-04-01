from __future__ import annotations

from tests.fixtures.operator_providers import build_attention_override_provider
from tests.fixtures.runtime_adapters import MockRuntimeAdapter
from vllm_xpu import bootstrap
from vllm_xpu.attention.backends.flash_attn import XPUAttentionImpl
from vllm_xpu.model_executor.layers.activation import apply_quick_gelu
from vllm_xpu.model_executor.layers.linear import apply_linear
from vllm_xpu.model_executor.layers.rotary_embedding import apply_rotary_embedding
from vllm_xpu.ops.registry import (
    clear_operator_providers,
    register_operator_provider,
    register_runtime_provider_override,
)
from vllm_xpu.runtime.registry import (
    activate_runtime_adapter,
    clear_runtime_adapters,
    register_runtime_adapter,
)


def setup_function() -> None:
    bootstrap._reset_for_tests()
    clear_runtime_adapters()
    clear_operator_providers()


def test_partial_provider_override_keeps_default_fallbacks() -> None:
    register_runtime_adapter("mock", MockRuntimeAdapter())
    activate_runtime_adapter("mock")
    bootstrap.initialize()

    register_operator_provider(
        "mock-attention-override",
        build_attention_override_provider(),
    )
    register_runtime_provider_override("mock", "mock-attention-override")

    attention = XPUAttentionImpl().forward("q", "k", "v", runtime_name="mock")
    linear = apply_linear("x", "w", "b", runtime_name="mock")
    quick_gelu = apply_quick_gelu("x", runtime_name="mock")
    rotary = apply_rotary_embedding("q", "k", "cos", "sin", runtime_name="mock")

    assert attention["provider"] == "mock-attention-override"
    assert linear["provider"] == "infinicore"
    assert quick_gelu["provider"] == "infinicore"
    assert rotary["provider"] == "infinicore"
