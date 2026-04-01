"""Built-in Infinicore runtime adapter."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from vllm_xpu.infinicore import get_infinicore
from vllm_xpu.runtime.base import BaseRuntimeAdapter


@dataclass
class InfinicoreRuntimeAdapter(BaseRuntimeAdapter):
    """Default runtime adapter backed by the vendored Infinicore package."""

    name: str = "infinicore"
    device_type_name: str = "xpu"
    dispatch_key_name: str = "XPU"
    preferred_device_types: tuple[str, ...] = ("mlu", "npu", "musa", "cuda", "cpu")

    def _module(self):
        return get_infinicore()

    def _runtime_device_type(self) -> str:
        override = os.environ.get("VLLM_XPU_DEVICE_TYPE")
        if override:
            return override

        module = self._module()
        if module is None:
            return self.device_type_name

        get_device_count = getattr(module, "get_device_count", None)
        if callable(get_device_count):
            for candidate in self.preferred_device_types:
                try:
                    if int(get_device_count(candidate)) > 0:
                        return candidate
                except Exception:
                    continue

        return self.device_type_name

    def is_available(self) -> bool:
        return self._module() is not None

    def device_type(self) -> str:
        return self._runtime_device_type()

    def dispatch_key(self) -> str:
        return self.dispatch_key_name

    def set_device(self, device: Any) -> None:
        module = self._module()
        if module is None:
            return None
        module.set_device(self._coerce_device(device))
        return None

    def synchronize(self) -> None:
        module = self._module()
        if module is None:
            return None
        module.sync_device()
        return None

    def empty_cache(self) -> None:
        # Infinicore does not currently expose a public empty-cache hook.
        return None

    def mem_get_info(self) -> tuple[int, int]:
        # Infinicore does not currently expose public memory-query APIs.
        return (0, 0)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"{self.device_type()}:{device_id}"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return 0

    def get_device_capability(self, device_id: int = 0) -> Any | None:
        return None

    def inference_mode(self):
        return nullcontext()

    def _coerce_device(self, device: Any):
        module = self._module()
        if module is None:
            return device
        if hasattr(device, "_underlying"):
            return device
        if isinstance(device, str):
            if ":" in device:
                device_type, device_index = device.split(":", 1)
                return module.device(device_type, int(device_index))
            return module.device(device)

        device_type = getattr(device, "type", self.device_type())
        device_index = getattr(device, "index", 0)
        return module.device(device_type, device_index)
