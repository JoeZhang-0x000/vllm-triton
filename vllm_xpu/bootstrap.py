"""Package bootstrap for vllm_xpu."""

from __future__ import annotations

from threading import Lock

_BOOTSTRAP_LOCK = Lock()
_BOOTSTRAPPED = False


def initialize() -> bool:
    """Initialize package-local registries once.

    Returns ``True`` if the current call performed initialization and
    ``False`` when the package was already initialized.
    """
    global _BOOTSTRAPPED
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return False

        from vllm_xpu.ops.registry import register_default_operator_providers
        from vllm_xpu.runtime.defaults import register_default_runtime_adapters

        register_default_runtime_adapters()
        register_default_operator_providers()
        _BOOTSTRAPPED = True
        return True


def is_initialized() -> bool:
    """Return whether package bootstrap has already run."""
    return _BOOTSTRAPPED


def _reset_for_tests() -> None:
    """Reset bootstrap state for unit tests."""
    global _BOOTSTRAPPED
    with _BOOTSTRAP_LOCK:
        _BOOTSTRAPPED = False

