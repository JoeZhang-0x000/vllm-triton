"""Operator provider fixtures for integration tests."""

from __future__ import annotations

from vllm_xpu.ops.provider import OperatorProvider


class _AttentionOverride:
    def forward(self, query, key, value, **kwargs):
        return {
            "provider": "mock-attention-override",
            "query": query,
            "key": key,
            "value": value,
            "kwargs": kwargs,
        }


def build_attention_override_provider() -> OperatorProvider:
    """Build a provider overriding only the attention group."""
    return OperatorProvider(
        name="mock-attention-override",
        implementations={"attention": _AttentionOverride()},
        metadata={"kind": "override"},
    )

