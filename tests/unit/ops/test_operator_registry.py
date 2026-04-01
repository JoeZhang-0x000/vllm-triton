from __future__ import annotations

import pytest

from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import (
    OperatorGroupNotSupported,
    clear_operator_providers,
    register_operator_provider,
    register_runtime_provider_override,
    resolve_operator,
    set_default_operator_provider,
)
from vllm_xpu.runtime.registry import (
    activate_runtime_adapter,
    clear_runtime_adapters,
    register_runtime_adapter,
)


class FakeRuntimeAdapter:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_type(self) -> str:
        return "mock"

    def dispatch_key(self) -> str:
        return "MOCK"

    def set_device(self, device: object) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (1, 2)

    def get_device_name(self, device_id: int = 0) -> str:
        return "mock-device"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 42

    def get_device_capability(self, device_id: int = 0) -> object | None:
        return None

    def inference_mode(self):
        return None


def setup_function() -> None:
    clear_runtime_adapters()
    clear_operator_providers()


def test_runtime_override_falls_back_to_default_provider_for_missing_groups() -> None:
    register_runtime_adapter("mock", FakeRuntimeAdapter(available=True))
    activate_runtime_adapter("mock")

    register_operator_provider(
        "default",
        OperatorProvider(
            name="default",
            implementations={
                "attention": "default-attention",
                "embedding": "default-embedding",
                "linear": "default-linear",
                "norm_act": "default-norm",
                "rotary": "default-rotary",
            },
        ),
        set_default=True,
    )
    register_operator_provider(
        "mock-override",
        OperatorProvider(
            name="mock-override",
            implementations={"attention": "override-attention"},
        ),
    )
    register_runtime_provider_override("mock", "mock-override")

    resolved_attention = resolve_operator("attention")
    resolved_embedding = resolve_operator("embedding")
    resolved_linear = resolve_operator("linear")

    assert resolved_attention.provider_name == "mock-override"
    assert resolved_attention.implementation == "override-attention"
    assert resolved_embedding.provider_name == "default"
    assert resolved_embedding.implementation == "default-embedding"
    assert resolved_linear.provider_name == "default"
    assert resolved_linear.implementation == "default-linear"


def test_resolve_operator_raises_clear_error_when_group_is_missing() -> None:
    register_operator_provider(
        "default",
        OperatorProvider(name="default", implementations={"attention": "a"}),
        set_default=True,
    )
    set_default_operator_provider("default")

    with pytest.raises(OperatorGroupNotSupported, match="linear"):
        resolve_operator("linear")
