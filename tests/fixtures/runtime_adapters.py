"""Runtime adapter test fixtures."""

from __future__ import annotations

from contextlib import nullcontext


class MockRuntimeAdapter:
    """Simple available runtime adapter used by integration tests."""

    def __init__(self, *, name: str = "mock", available: bool = True) -> None:
        self.name = name
        self.available = available
        self.devices: list[object] = []

    def is_available(self) -> bool:
        return self.available

    def device_type(self) -> str:
        return self.name

    def dispatch_key(self) -> str:
        return self.name.upper()

    def set_device(self, device: object) -> None:
        self.devices.append(device)

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def mem_get_info(self) -> tuple[int, int]:
        return (1024, 2048)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"{self.name}:{device_id}"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 2048

    def get_device_capability(self, device_id: int = 0):
        return {"major": 1, "minor": 0}

    def inference_mode(self):
        return nullcontext()

