"""Registry for runtime adapters."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock

from vllm_xpu.runtime.base import RuntimeAdapter


class RuntimeAdapterAlreadyRegistered(RuntimeError):
    """Raised when a runtime adapter name is already registered."""


class RuntimeAdapterNotFound(LookupError):
    """Raised when a runtime adapter cannot be found."""


class RuntimeAdapterUnavailable(RuntimeError):
    """Raised when the selected runtime adapter is unavailable."""


class NoRuntimeAvailable(RuntimeError):
    """Raised when no available runtime adapter can be resolved."""


_RUNTIME_ADAPTERS: "OrderedDict[str, RuntimeAdapter]" = OrderedDict()
_ACTIVE_RUNTIME_NAME: str | None = None
_LOCK = RLock()


def register_runtime_adapter(
    name: str,
    adapter: RuntimeAdapter,
    *,
    replace: bool = False,
) -> RuntimeAdapter:
    """Register a named runtime adapter."""
    if not name:
        raise ValueError("runtime adapter name must be non-empty")
    with _LOCK:
        if name in _RUNTIME_ADAPTERS and not replace:
            raise RuntimeAdapterAlreadyRegistered(
                f"runtime adapter {name!r} is already registered"
            )
        _RUNTIME_ADAPTERS[name] = adapter
        return adapter


def clear_runtime_adapters() -> None:
    """Clear runtime adapter registry state."""
    global _ACTIVE_RUNTIME_NAME
    with _LOCK:
        _RUNTIME_ADAPTERS.clear()
        _ACTIVE_RUNTIME_NAME = None


def list_runtime_adapters() -> tuple[str, ...]:
    """Return the registered runtime adapter names in registration order."""
    with _LOCK:
        return tuple(_RUNTIME_ADAPTERS.keys())


def get_runtime_adapter(name: str) -> RuntimeAdapter:
    """Return a runtime adapter by name."""
    with _LOCK:
        try:
            return _RUNTIME_ADAPTERS[name]
        except KeyError as exc:
            raise RuntimeAdapterNotFound(
                f"runtime adapter {name!r} is not registered"
            ) from exc


def activate_runtime_adapter(name: str) -> RuntimeAdapter:
    """Mark a runtime adapter as active."""
    global _ACTIVE_RUNTIME_NAME
    adapter = get_runtime_adapter(name)
    with _LOCK:
        _ACTIVE_RUNTIME_NAME = name
    return adapter


def get_active_runtime_name() -> str | None:
    """Return the explicitly activated runtime adapter name, if any."""
    with _LOCK:
        return _ACTIVE_RUNTIME_NAME


def get_active_runtime_adapter(
    *,
    require_available: bool = False,
) -> RuntimeAdapter | None:
    """Return the explicitly active runtime adapter, if any."""
    active_name = get_active_runtime_name()
    if active_name is None:
        return None
    adapter = get_runtime_adapter(active_name)
    if require_available and not adapter.is_available():
        raise RuntimeAdapterUnavailable(
            f"active runtime adapter {active_name!r} is not available"
        )
    return adapter


def find_available_runtime_adapter() -> tuple[str, RuntimeAdapter] | None:
    """Return the first available runtime adapter in registration order."""
    with _LOCK:
        items = tuple(_RUNTIME_ADAPTERS.items())
    for name, adapter in items:
        if adapter.is_available():
            return name, adapter
    return None


def has_available_runtime_adapter() -> bool:
    """Return whether any registered runtime adapter is currently available."""
    return find_available_runtime_adapter() is not None


def resolve_runtime_adapter(
    *,
    require_available: bool = True,
) -> tuple[str, RuntimeAdapter]:
    """Resolve the runtime adapter to use.

    Resolution order:
    1. Explicitly active runtime adapter
    2. First registered available runtime adapter
    """
    active_name = get_active_runtime_name()
    if active_name is not None:
        adapter = get_runtime_adapter(active_name)
        if require_available and not adapter.is_available():
            raise RuntimeAdapterUnavailable(
                f"active runtime adapter {active_name!r} is not available"
            )
        return active_name, adapter

    available = find_available_runtime_adapter()
    if available is not None:
        return available

    if require_available:
        raise NoRuntimeAvailable("no available runtime adapter is registered")

    with _LOCK:
        if _RUNTIME_ADAPTERS:
            name, adapter = next(iter(_RUNTIME_ADAPTERS.items()))
            return name, adapter
    raise NoRuntimeAvailable("no runtime adapter is registered")

