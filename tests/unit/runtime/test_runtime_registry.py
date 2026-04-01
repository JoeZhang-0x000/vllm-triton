from __future__ import annotations

from contextlib import nullcontext

import pytest

from vllm_xpu.runtime.base import BaseRuntimeAdapter
from vllm_xpu.runtime.registry import (
    NoRuntimeAvailable,
    RuntimeAdapterAlreadyRegistered,
    RuntimeAdapterUnavailable,
    activate_runtime_adapter,
    clear_runtime_adapters,
    find_available_runtime_adapter,
    get_active_runtime_adapter,
    list_runtime_adapters,
    register_runtime_adapter,
    resolve_runtime_adapter,
)


class FakeRuntimeAdapter(BaseRuntimeAdapter):
    def __init__(self, name: str, available: bool) -> None:
        self.name = name
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def device_type(self) -> str:
        return self.name

    def dispatch_key(self) -> str:
        return self.name.upper()

    def set_device(self, device: object) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (1, 2)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"{self.name}:{device_id}"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 42

    def get_device_capability(self, device_id: int = 0) -> object | None:
        return {"device_id": device_id}

    def inference_mode(self):
        return nullcontext()


def setup_function() -> None:
    clear_runtime_adapters()


def test_register_runtime_adapter_rejects_duplicate_names() -> None:
    register_runtime_adapter("mock", FakeRuntimeAdapter("mock", available=True))

    with pytest.raises(RuntimeAdapterAlreadyRegistered):
        register_runtime_adapter("mock", FakeRuntimeAdapter("mock", available=True))


def test_resolve_runtime_adapter_picks_first_available_when_no_active_runtime() -> None:
    register_runtime_adapter("unavailable", FakeRuntimeAdapter("u", available=False))
    register_runtime_adapter("available", FakeRuntimeAdapter("a", available=True))

    name, adapter = resolve_runtime_adapter()

    assert name == "available"
    assert adapter.device_type() == "a"
    assert find_available_runtime_adapter()[0] == "available"
    assert list_runtime_adapters() == ("unavailable", "available")


def test_active_runtime_must_be_available_when_required() -> None:
    register_runtime_adapter("mock", FakeRuntimeAdapter("mock", available=False))
    activate_runtime_adapter("mock")

    with pytest.raises(RuntimeAdapterUnavailable):
        get_active_runtime_adapter(require_available=True)

    with pytest.raises(RuntimeAdapterUnavailable):
        resolve_runtime_adapter()


def test_resolve_runtime_adapter_raises_when_no_runtime_is_registered() -> None:
    with pytest.raises(NoRuntimeAvailable):
        resolve_runtime_adapter()

