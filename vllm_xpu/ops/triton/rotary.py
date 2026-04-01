"""Default rotary operator implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TritonRotaryOps:
    """Default rotary implementation placeholder."""

    provider_name: str = "triton"

    def apply(
        self,
        query: Any,
        key: Any,
        cos: Any,
        sin: Any,
        *,
        position_ids: Any | None = None,
    ) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "query": query,
            "key": key,
            "cos": cos,
            "sin": sin,
            "position_ids": position_ids,
        }

