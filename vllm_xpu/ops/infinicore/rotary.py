"""Default rotary implementation backed by Infinicore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_xpu.ops.infinicore._base import InfinicoreOpMixin


@dataclass
class InfinicoreRotaryOps(InfinicoreOpMixin):
    provider_name: str = "infinicore"

    def apply(
        self,
        query: Any,
        key: Any,
        cos: Any,
        sin: Any,
        *,
        position_ids: Any | None = None,
    ) -> Any:
        module = self._module()
        if (
            module is not None
            and position_ids is not None
            and self._is_infinicore_tensor(query)
            and self._is_infinicore_tensor(key)
            and self._is_infinicore_tensor(cos)
            and self._is_infinicore_tensor(sin)
            and self._is_infinicore_tensor(position_ids)
        ):
            rotated_query = module.nn.functional.rope(query, position_ids, sin, cos)
            rotated_key = module.nn.functional.rope(key, position_ids, sin, cos)
            return rotated_query, rotated_key

        return {
            "provider": self.provider_name,
            "op": "rotary",
            "query": query,
            "key": key,
            "cos": cos,
            "sin": sin,
            "position_ids": position_ids,
        }
