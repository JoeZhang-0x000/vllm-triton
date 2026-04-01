"""Default linear operator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TritonLinearOps:
    """Default dense linear implementation with a torch-first fallback."""

    provider_name: str = "triton"

    def forward(self, x: Any, weight: Any, bias: Any | None = None) -> Any:
        try:
            output = x @ weight.transpose(-1, -2)
        except Exception:
            output = {
                "provider": self.provider_name,
                "x": x,
                "weight": weight,
                "bias": bias,
            }
            return output

        if bias is not None:
            output = output + bias
        return output

