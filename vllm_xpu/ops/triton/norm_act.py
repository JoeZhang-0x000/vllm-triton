"""Default norm and activation implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TritonNormActOps:
    """Default norm/activation implementation with torch-first behavior."""

    provider_name: str = "triton"

    def rms_norm(self, x: Any, weight: Any, eps: float = 1e-6) -> Any:
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

