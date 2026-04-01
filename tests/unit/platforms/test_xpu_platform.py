from __future__ import annotations

from contextlib import nullcontext

from vllm_xpu.platforms.xpu import XPUPlatform
from vllm_xpu.plugins import register_xpu_platform
from vllm_xpu.runtime.registry import (
    activate_runtime_adapter,
    clear_runtime_adapters,
    register_runtime_adapter,
)


class FakeRuntimeAdapter:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.set_device_calls = []

    def is_available(self) -> bool:
        return self.available

    def device_type(self) -> str:
        return "mock"

    def dispatch_key(self) -> str:
        return "MOCK"

    def set_device(self, device: object) -> None:
        self.set_device_calls.append(device)

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (10, 20)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"mock:{device_id}"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 20

    def get_device_capability(self, device_id: int = 0):
        return {"major": 1, "minor": 0}

    def inference_mode(self):
        return nullcontext()


def setup_function() -> None:
    clear_runtime_adapters()


def test_platform_plugin_activates_only_when_runtime_is_available() -> None:
    assert register_xpu_platform() is None

    register_runtime_adapter("mock", FakeRuntimeAdapter(available=True))

    assert register_xpu_platform() == "vllm_xpu.platforms.xpu.XPUPlatform"


def test_xpu_platform_delegates_device_methods_to_runtime_adapter() -> None:
    adapter = FakeRuntimeAdapter(available=True)
    register_runtime_adapter("mock", adapter)
    activate_runtime_adapter("mock")

    XPUPlatform.sync_runtime_metadata()
    XPUPlatform.set_device("mock:0")

    assert adapter.set_device_calls == ["mock:0"]
    assert XPUPlatform.device_type == "mock"
    assert XPUPlatform.dispatch_key == "MOCK"
    assert XPUPlatform.mem_get_info() == (10, 20)
    assert XPUPlatform.get_device_name() == "mock:0"
    assert XPUPlatform.get_device_uuid() == "mock:0"
    assert XPUPlatform.get_device_total_memory() == 20
    assert XPUPlatform.get_device_capability() == {"major": 1, "minor": 0}
    assert XPUPlatform.get_attn_backend_cls(None, 128, None, None, 16, True, False) == (
        "vllm_xpu.attention.backends.flash_attn.XPUAttentionBackend"
    )
