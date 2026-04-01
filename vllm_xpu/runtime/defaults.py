"""Default runtime registrations for vllm_xpu."""

from __future__ import annotations


def register_default_runtime_adapters() -> None:
    """Register built-in runtime adapters.

    The foundation phase intentionally ships no built-in device runtime.
    Concrete runtimes will be registered by later device-specific modules.
    """
    return None

