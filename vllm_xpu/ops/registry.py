"""Registry for operator providers and runtime-specific overrides."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from vllm_xpu.ops.groups import OperatorGroup, normalize_operator_group
from vllm_xpu.ops.provider import OperatorProvider


class OperatorProviderAlreadyRegistered(RuntimeError):
    """Raised when an operator provider name is already registered."""


class OperatorProviderNotFound(LookupError):
    """Raised when an operator provider cannot be found."""


class OperatorGroupNotSupported(LookupError):
    """Raised when no provider can satisfy the requested operator group."""


@dataclass(frozen=True)
class ResolvedOperator:
    """Resolved operator implementation with provider metadata."""

    group: OperatorGroup
    implementation: object
    provider_name: str
    provider: OperatorProvider


_OPERATOR_PROVIDERS: "OrderedDict[str, OperatorProvider]" = OrderedDict()
_DEFAULT_PROVIDER_NAME: str | None = None
_RUNTIME_PROVIDER_OVERRIDES: dict[str, tuple[str, ...]] = {}
_LOCK = RLock()


def register_operator_provider(
    name: str,
    provider: OperatorProvider,
    *,
    replace: bool = False,
    set_default: bool = False,
) -> OperatorProvider:
    """Register a named operator provider."""
    global _DEFAULT_PROVIDER_NAME
    if not name:
        raise ValueError("operator provider name must be non-empty")
    with _LOCK:
        if name in _OPERATOR_PROVIDERS and not replace:
            raise OperatorProviderAlreadyRegistered(
                f"operator provider {name!r} is already registered"
            )
        _OPERATOR_PROVIDERS[name] = provider
        if set_default or _DEFAULT_PROVIDER_NAME is None:
            _DEFAULT_PROVIDER_NAME = name
        return provider


def clear_operator_providers() -> None:
    """Clear operator provider registry state."""
    global _DEFAULT_PROVIDER_NAME
    with _LOCK:
        _OPERATOR_PROVIDERS.clear()
        _RUNTIME_PROVIDER_OVERRIDES.clear()
        _DEFAULT_PROVIDER_NAME = None


def register_default_operator_providers() -> None:
    """Register built-in operator providers.

    Register the package-default Triton provider exactly once.
    """
    from vllm_xpu.ops.providers.triton_provider import (
        TRITON_PROVIDER_NAME,
        build_triton_provider,
    )

    with _LOCK:
        if TRITON_PROVIDER_NAME in _OPERATOR_PROVIDERS:
            return

    register_operator_provider(
        TRITON_PROVIDER_NAME,
        build_triton_provider(),
        set_default=True,
    )


def list_operator_providers() -> tuple[str, ...]:
    """Return registered operator provider names in registration order."""
    with _LOCK:
        return tuple(_OPERATOR_PROVIDERS.keys())


def get_operator_provider(name: str) -> OperatorProvider:
    """Return an operator provider by name."""
    with _LOCK:
        try:
            return _OPERATOR_PROVIDERS[name]
        except KeyError as exc:
            raise OperatorProviderNotFound(
                f"operator provider {name!r} is not registered"
            ) from exc


def set_default_operator_provider(name: str) -> None:
    """Set the default operator provider."""
    global _DEFAULT_PROVIDER_NAME
    get_operator_provider(name)
    with _LOCK:
        _DEFAULT_PROVIDER_NAME = name


def get_default_operator_provider() -> OperatorProvider | None:
    """Return the default operator provider, if configured."""
    with _LOCK:
        name = _DEFAULT_PROVIDER_NAME
    if name is None:
        return None
    return get_operator_provider(name)


def register_runtime_provider_override(
    runtime_name: str,
    provider_name: str,
) -> None:
    """Associate a runtime with a higher-priority provider override."""
    get_operator_provider(provider_name)
    with _LOCK:
        existing = _RUNTIME_PROVIDER_OVERRIDES.get(runtime_name, ())
        if provider_name in existing:
            return
        _RUNTIME_PROVIDER_OVERRIDES[runtime_name] = existing + (provider_name,)


def get_runtime_provider_overrides(runtime_name: str) -> tuple[str, ...]:
    """Return provider overrides for a runtime in priority order."""
    with _LOCK:
        return _RUNTIME_PROVIDER_OVERRIDES.get(runtime_name, ())


def resolve_operator(
    group: OperatorGroup | str,
    *,
    runtime_name: str | None = None,
) -> ResolvedOperator:
    """Resolve an operator implementation for a group.

    Resolution order:
    1. Runtime-specific provider overrides
    2. Default provider
    """
    normalized = normalize_operator_group(group)

    if runtime_name is None:
        from vllm_xpu.runtime.registry import get_active_runtime_name

        runtime_name = get_active_runtime_name()

    candidate_names: list[str] = []
    if runtime_name is not None:
        candidate_names.extend(get_runtime_provider_overrides(runtime_name))
    with _LOCK:
        if _DEFAULT_PROVIDER_NAME is not None:
            candidate_names.append(_DEFAULT_PROVIDER_NAME)

    seen: set[str] = set()
    for provider_name in candidate_names:
        if provider_name in seen:
            continue
        seen.add(provider_name)
        provider = get_operator_provider(provider_name)
        if provider.supports(normalized):
            return ResolvedOperator(
                group=normalized,
                implementation=provider.get(normalized),
                provider_name=provider_name,
                provider=provider,
            )

    runtime_label = runtime_name or "<none>"
    raise OperatorGroupNotSupported(
        f"operator group {normalized.value!r} is not supported for runtime "
        f"{runtime_label!r}"
    )
