"""Best-effort registration hooks for integrating with upstream vLLM."""

from __future__ import annotations

from threading import Lock

from vllm_xpu.worker.worker import DEFAULT_XPU_ATTN_BACKEND

_LOCK = Lock()
_REGISTERED = False


def register_with_vllm() -> bool:
    """Register custom attention/backend overrides when vLLM is importable."""
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return False

        try:
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
                register_backend,
            )
        except ImportError:
            return False

        register_backend(AttentionBackendEnum.CUSTOM, DEFAULT_XPU_ATTN_BACKEND)

        try:
            import vllm_xpu.oot  # noqa: F401
        except ImportError:
            return False

        _REGISTERED = True
        return True


def _reset_for_tests() -> None:
    """Reset registration state for unit tests."""
    global _REGISTERED
    with _LOCK:
        _REGISTERED = False
