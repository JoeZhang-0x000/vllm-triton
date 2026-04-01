"""Operator provider objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from vllm_xpu.ops.groups import OperatorGroup, normalize_operator_group


@dataclass(frozen=True)
class OperatorProvider:
    """A named set of operator-group implementations."""

    name: str
    implementations: Mapping[OperatorGroup | str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def supported_groups(self) -> frozenset[OperatorGroup]:
        """Return the groups implemented by this provider."""
        return frozenset(
            normalize_operator_group(group) for group in self.implementations
        )

    def supports(self, group: OperatorGroup | str) -> bool:
        """Return whether the provider implements a group."""
        normalized = normalize_operator_group(group)
        return normalized in self.supported_groups()

    def get(self, group: OperatorGroup | str) -> Any:
        """Return the implementation object for a group."""
        normalized = normalize_operator_group(group)
        for key, implementation in self.implementations.items():
            if normalize_operator_group(key) == normalized:
                return implementation
        raise KeyError(normalized.value)

