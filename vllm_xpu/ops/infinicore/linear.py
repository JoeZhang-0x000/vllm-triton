"""Default dense linear implementation backed by Infinicore when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_xpu.ops.infinicore._base import InfinicoreOpMixin


@dataclass
class InfinicoreLinearOps(InfinicoreOpMixin):
    provider_name: str = "infinicore"

    def forward(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        module = self._module()
        if (
            module is not None
            and self._is_infinicore_tensor(x)
            and self._is_infinicore_tensor(weight)
            and (bias is None or self._is_infinicore_tensor(bias))
        ):
            return module.nn.functional.linear(x, weight, bias)

        try:
            output = x @ weight.transpose(-1, -2)
        except Exception:
            return {
                "provider": self.provider_name,
                "op": "linear",
                "x": x,
                "weight": weight,
                "bias": bias,
            }

        if bias is not None:
            output = output + bias
        return output
