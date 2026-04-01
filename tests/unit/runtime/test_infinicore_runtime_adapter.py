from __future__ import annotations

from types import SimpleNamespace

from vllm_xpu.runtime.infinicore import InfinicoreRuntimeAdapter


class FakeInfinicoreModule:
    def __init__(self) -> None:
        self.devices = []
        self.sync_calls = 0

    def device(self, device_type, index=0):
        return SimpleNamespace(type=device_type, index=index)

    def set_device(self, device) -> None:
        self.devices.append(device)

    def sync_device(self) -> None:
        self.sync_calls += 1


def test_infinicore_runtime_adapter_uses_module_when_available(monkeypatch) -> None:
    module = FakeInfinicoreModule()
    monkeypatch.setattr(
        "vllm_xpu.runtime.infinicore.get_infinicore",
        lambda: module,
    )

    adapter = InfinicoreRuntimeAdapter()
    device = adapter._coerce_device("xpu:2")
    adapter.set_device(device)
    adapter.synchronize()

    assert adapter.is_available() is True
    assert adapter.device_type() == "xpu"
    assert adapter.dispatch_key() == "XPU"
    assert module.devices[0].type == "xpu"
    assert module.devices[0].index == 2
    assert module.sync_calls == 1


def test_infinicore_runtime_adapter_is_unavailable_without_module(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_xpu.runtime.infinicore.get_infinicore",
        lambda: None,
    )

    adapter = InfinicoreRuntimeAdapter()

    assert adapter.is_available() is False
    assert adapter.mem_get_info() == (0, 0)
    assert adapter.get_device_name(3) == "xpu:3"
