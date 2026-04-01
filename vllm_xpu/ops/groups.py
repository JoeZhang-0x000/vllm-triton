"""Stable operator-group identifiers."""

from __future__ import annotations

from enum import Enum


class OperatorGroup(str, Enum):
    """Stable operator groups supported by the XPU framework."""

    ATTENTION = "attention"
    EMBEDDING = "embedding"
    LINEAR = "linear"
    NORM_ACT = "norm_act"
    ROTARY = "rotary"


def normalize_operator_group(value: OperatorGroup | str) -> OperatorGroup:
    """Normalize a string or enum value into an ``OperatorGroup``."""
    if isinstance(value, OperatorGroup):
        return value
    return OperatorGroup(value)
