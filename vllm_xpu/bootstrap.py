"""Package bootstrap for vllm_xpu."""

from __future__ import annotations

from threading import RLock

_BOOTSTRAP_LOCK = RLock()
_BOOTSTRAPPED = False
_BOOTSTRAPPING = False


def initialize() -> bool:
    """Initialize package-local registries once.

    Returns ``True`` if the current call performed initialization and
    ``False`` when the package was already initialized.
    """
    global _BOOTSTRAPPED, _BOOTSTRAPPING
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return False
        if _BOOTSTRAPPING:
            return False
        _BOOTSTRAPPING = True

    try:
        from vllm_xpu.ops.registry import register_default_operator_providers
        from vllm_xpu.runtime.defaults import register_default_runtime_adapters
        from vllm_xpu.vllm_registration import register_with_vllm

        register_default_runtime_adapters()
        register_default_operator_providers()
        register_with_vllm()
        with _BOOTSTRAP_LOCK:
            _BOOTSTRAPPED = True
        return True
    finally:
        with _BOOTSTRAP_LOCK:
            _BOOTSTRAPPING = False


def is_initialized() -> bool:
    """Return whether package bootstrap has already run."""
    return _BOOTSTRAPPED


def _reset_for_tests() -> None:
    """Reset bootstrap state for unit tests."""
    global _BOOTSTRAPPED, _BOOTSTRAPPING
    with _BOOTSTRAP_LOCK:
        _BOOTSTRAPPED = False
        _BOOTSTRAPPING = False
