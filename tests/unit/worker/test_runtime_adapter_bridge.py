from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import clear_operator_providers, register_operator_provider
from vllm_xpu.runtime.registry import (
    clear_runtime_adapters,
    register_runtime_adapter,
)
from vllm_xpu.worker.worker import XPUWorker


class FakeRuntimeAdapter:
    def __init__(self) -> None:
        self.devices = []

    def is_available(self) -> bool:
        return True

    def device_type(self) -> str:
        return "mock"

    def dispatch_key(self) -> str:
        return "MOCK"

    def set_device(self, device: object) -> None:
        self.devices.append(device)

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (7, 11)

    def get_device_name(self, device_id: int = 0) -> str:
        return "mock:0"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 11

    def get_device_capability(self, device_id: int = 0):
        return None

    def inference_mode(self):
        return nullcontext()


def setup_function() -> None:
    clear_runtime_adapters()
    clear_operator_providers()


def test_worker_uses_runtime_adapter_and_operator_bridge() -> None:
    adapter = FakeRuntimeAdapter()
    register_runtime_adapter("mock", adapter)
    register_operator_provider(
        "default",
        OperatorProvider(name="default", implementations={"attention": "impl"}),
        set_default=True,
    )

    worker = XPUWorker(vllm_config=SimpleNamespace(), local_rank=0)
    device = worker.init_device()
    snapshot = worker.get_memory_snapshot()
    resolved = worker.model_runner.resolve_operator_group("attention")

    assert str(device).endswith("0")
    assert adapter.devices
    assert snapshot.runtime_name == "mock"
    assert snapshot.free_memory == 7
    assert resolved.provider_name == "default"
    assert resolved.implementation == "impl"
