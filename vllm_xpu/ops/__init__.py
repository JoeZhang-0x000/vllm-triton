"""Operator provider contracts and registry helpers."""

from vllm_xpu.ops.groups import OperatorGroup, normalize_operator_group
from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import (
    OperatorGroupNotSupported,
    OperatorProviderAlreadyRegistered,
    OperatorProviderNotFound,
    ResolvedOperator,
    clear_operator_providers,
    get_default_operator_provider,
    get_operator_provider,
    get_runtime_provider_overrides,
    list_operator_providers,
    register_default_operator_providers,
    register_operator_provider,
    register_runtime_provider_override,
    resolve_operator,
    set_default_operator_provider,
)

__all__ = [
    "OperatorGroup",
    "normalize_operator_group",
    "OperatorProvider",
    "OperatorGroupNotSupported",
    "OperatorProviderAlreadyRegistered",
    "OperatorProviderNotFound",
    "ResolvedOperator",
    "clear_operator_providers",
    "get_default_operator_provider",
    "get_operator_provider",
    "get_runtime_provider_overrides",
    "list_operator_providers",
    "register_default_operator_providers",
    "register_operator_provider",
    "register_runtime_provider_override",
    "resolve_operator",
    "set_default_operator_provider",
]

