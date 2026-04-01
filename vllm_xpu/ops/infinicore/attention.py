"""Default attention implementation backed by Infinicore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_xpu.ops.infinicore._base import InfinicoreOpMixin


@dataclass
class InfinicoreAttentionOps(InfinicoreOpMixin):
    provider_name: str = "infinicore"

    def forward(self, query: Any, key: Any, value: Any, **kwargs: Any) -> Any:
        module = self._module()
        if module is not None and self._is_infinicore_tensor(query):
            block_tables = kwargs.get("block_tables") or kwargs.get("block_table")
            scale = kwargs.get("scale", 1.0)
            alibi_slopes = kwargs.get("alibi_slopes")

            if (
                self._is_infinicore_tensor(kwargs.get("k_cache"))
                and self._is_infinicore_tensor(kwargs.get("v_cache"))
                and self._is_infinicore_tensor(block_tables)
                and self._is_infinicore_tensor(kwargs.get("history_lens"))
                and self._is_infinicore_tensor(kwargs.get("cu_seqlens_q"))
            ):
                return module.paged_attention_prefill(
                    query,
                    kwargs["k_cache"],
                    kwargs["v_cache"],
                    block_tables,
                    kwargs["history_lens"],
                    kwargs["cu_seqlens_q"],
                    alibi_slopes,
                    scale,
                )

            if (
                self._is_infinicore_tensor(kwargs.get("k_cache"))
                and self._is_infinicore_tensor(kwargs.get("v_cache"))
                and self._is_infinicore_tensor(block_tables)
                and self._is_infinicore_tensor(kwargs.get("cache_lens"))
            ):
                return module.paged_attention(
                    query,
                    kwargs["k_cache"],
                    kwargs["v_cache"],
                    block_tables,
                    kwargs["cache_lens"],
                    alibi_slopes,
                    scale,
                )

            if (
                self._is_infinicore_tensor(key)
                and self._is_infinicore_tensor(value)
                and self._is_infinicore_tensor(kwargs.get("k_cache"))
                and self._is_infinicore_tensor(kwargs.get("v_cache"))
                and "pos" in kwargs
            ):
                return module.attention(
                    query,
                    key,
                    value,
                    kwargs["k_cache"],
                    kwargs["v_cache"],
                    kwargs["pos"],
                )

        return {
            "provider": self.provider_name,
            "op": "attention",
            "query": query,
            "key": key,
            "value": value,
            "kwargs": kwargs,
        }
