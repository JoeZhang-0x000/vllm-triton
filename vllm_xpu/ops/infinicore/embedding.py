"""Default embedding implementation backed by Infinicore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_xpu.ops.infinicore._base import InfinicoreOpMixin


@dataclass
class InfinicoreEmbeddingOps(InfinicoreOpMixin):
    provider_name: str = "infinicore"

    def forward(self, indices: Any, weight: Any) -> Any:
        module = self._module()
        if (
            module is not None
            and self._is_infinicore_tensor(indices)
            and self._is_infinicore_tensor(weight)
        ):
            return module.nn.functional.embedding(indices, weight)

        return {
            "provider": self.provider_name,
            "op": "embedding",
            "indices": indices,
            "weight": weight,
        }
