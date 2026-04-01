"""Shared helpers for Infinicore-backed operator implementations."""

from __future__ import annotations

from typing import Any

from vllm_xpu.infinicore import get_infinicore


class InfinicoreOpMixin:
    provider_name: str = "infinicore"

    def _module(self):
        return get_infinicore()

    @staticmethod
    def _is_infinicore_tensor(value: Any) -> bool:
        return hasattr(value, "_underlying")
