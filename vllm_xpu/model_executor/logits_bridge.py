"""Helpers for exporting device logits back to the vLLM host boundary."""

from __future__ import annotations

from typing import Any


def export_logits_to_host(logits: Any) -> Any:
    """Return a host-consumable logits object when device tensors need bridging."""
    to_host = getattr(logits, "to_host", None)
    if callable(to_host):
        return to_host()
    return logits
