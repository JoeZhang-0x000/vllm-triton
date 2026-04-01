from __future__ import annotations

from vllm_xpu.attention.backends.flash_attn import XPUAttentionImpl
from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import clear_operator_providers, register_operator_provider
from vllm_xpu.runtime.registry import clear_runtime_adapters


def setup_function() -> None:
    clear_runtime_adapters()
    clear_operator_providers()


def test_attention_impl_dispatches_through_operator_registry() -> None:
    register_operator_provider(
        "default",
        OperatorProvider(
            name="default",
            implementations={
                "attention": type(
                    "AttentionImpl",
                    (),
                    {
                        "forward": staticmethod(
                            lambda q, k, v, **kwargs: {
                                "q": q,
                                "k": k,
                                "v": v,
                                "kwargs": kwargs,
                            }
                        )
                    },
                )()
            },
        ),
        set_default=True,
    )

    output = XPUAttentionImpl().forward("q", "k", "v", mask="m")

    assert output["q"] == "q"
    assert output["k"] == "k"
    assert output["v"] == "v"
    assert output["kwargs"] == {"mask": "m"}

