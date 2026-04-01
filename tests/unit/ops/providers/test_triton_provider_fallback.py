from __future__ import annotations

from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import (
    clear_operator_providers,
    register_default_operator_providers,
    register_operator_provider,
    register_runtime_provider_override,
    resolve_operator,
)
from vllm_xpu.runtime.registry import (
    activate_runtime_adapter,
    clear_runtime_adapters,
    register_runtime_adapter,
)


class FakeRuntimeAdapter:
    def is_available(self) -> bool:
        return True

    def device_type(self) -> str:
        return "mock"

    def dispatch_key(self) -> str:
        return "MOCK"

    def set_device(self, device):
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (1, 2)

    def get_device_name(self, device_id: int = 0) -> str:
        return "mock:0"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 2

    def get_device_capability(self, device_id: int = 0):
        return None

    def inference_mode(self):
        return None


def setup_function() -> None:
    clear_runtime_adapters()
    clear_operator_providers()


def test_triton_provider_is_registered_as_default_and_used_as_fallback() -> None:
    register_default_operator_providers()
    register_runtime_adapter("mock", FakeRuntimeAdapter())
    activate_runtime_adapter("mock")
    register_operator_provider(
        "mock-override",
        OperatorProvider(name="mock-override", implementations={"attention": "override"}),
    )
    register_runtime_provider_override("mock", "mock-override")

    resolved_attention = resolve_operator("attention")
    resolved_linear = resolve_operator("linear")
    resolved_norm_act = resolve_operator("norm_act")
    resolved_rotary = resolve_operator("rotary")

    assert resolved_attention.provider_name == "mock-override"
    assert resolved_linear.provider_name == "triton"
    assert resolved_norm_act.provider_name == "triton"
    assert resolved_rotary.provider_name == "triton"
