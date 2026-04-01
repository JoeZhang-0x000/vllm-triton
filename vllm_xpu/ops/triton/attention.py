"""Default attention operator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TritonAttentionOps:
    """Default attention implementation placeholder."""

    provider_name: str = "triton"

    def forward(self, query: Any, key: Any, value: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "query": query,
            "key": key,
            "value": value,
            "kwargs": kwargs,
        }

