"""Base runtime adapter contracts for vllm_xpu."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Minimal runtime contract required by the XPU framework."""

    def is_available(self) -> bool: ...

    def device_type(self) -> str: ...

    def dispatch_key(self) -> str: ...

    def set_device(self, device: Any) -> None: ...

    def synchronize(self) -> None: ...

    def empty_cache(self) -> None: ...

    def mem_get_info(self) -> tuple[int, int]: ...

    def get_device_name(self, device_id: int = 0) -> str: ...

    def get_device_total_memory(self, device_id: int = 0) -> int: ...

    def get_device_capability(self, device_id: int = 0) -> Any | None: ...

    def inference_mode(self) -> AbstractContextManager[Any]: ...


class BaseRuntimeAdapter(ABC):
    """ABC form of the runtime adapter contract."""

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def device_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def dispatch_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def set_device(self, device: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def empty_cache(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def mem_get_info(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def get_device_name(self, device_id: int = 0) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_device_total_memory(self, device_id: int = 0) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_device_capability(self, device_id: int = 0) -> Any | None:
        raise NotImplementedError

    def inference_mode(self) -> AbstractContextManager[Any]:
        return nullcontext()

