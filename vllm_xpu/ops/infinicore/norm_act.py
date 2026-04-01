"""Default norm and activation implementations backed by Infinicore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_xpu.ops.infinicore._base import InfinicoreOpMixin


@dataclass
class InfinicoreNormActOps(InfinicoreOpMixin):
    provider_name: str = "infinicore"

    def rms_norm(self, x: Any, weight: Any, eps: float = 1e-6) -> Any:
        module = self._module()
        if (
            module is not None
            and self._is_infinicore_tensor(x)
            and self._is_infinicore_tensor(weight)
        ):
            normalized_shape = list(getattr(weight, "shape", [])) or [x.shape[-1]]
            return module.nn.functional.rms_norm(x, normalized_shape, weight, eps)

        try:
            import torch

            variance = x.pow(2).mean(dim=-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + eps)
            return normalized * weight
        except Exception:
            return {
                "provider": self.provider_name,
                "op": "rms_norm",
                "x": x,
                "weight": weight,
                "eps": eps,
            }

    def silu_mul(self, x: Any, gate: Any) -> Any:
        module = self._module()
        if (
            module is not None
            and self._is_infinicore_tensor(x)
            and self._is_infinicore_tensor(gate)
        ):
            return module.nn.functional.swiglu(x, gate)

        try:
            import torch.nn.functional as F

            return F.silu(x) * gate
        except Exception:
            return {
                "provider": self.provider_name,
                "op": "silu_mul",
                "x": x,
                "gate": gate,
            }

    def quick_gelu(self, x: Any) -> Any:
        try:
            import torch

            return x * torch.sigmoid(1.702 * x)
        except Exception:
            return {
                "provider": self.provider_name,
                "op": "quick_gelu",
                "x": x,
            }
