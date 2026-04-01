from __future__ import annotations

from types import SimpleNamespace

from vllm_xpu.model_executor.conversion import to_infinicore_tensor


class FakeInfinicoreModule:
    def from_list(self, value, dtype=None, device=None):
        return SimpleNamespace(kind="tensor", value=value, dtype=dtype, device=device)


def test_to_infinicore_tensor_converts_lists_when_module_is_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_xpu.model_executor.conversion.get_infinicore",
        lambda: FakeInfinicoreModule(),
    )

    tensor = to_infinicore_tensor([1, 2, 3], dtype="bf16", device="xpu:0")

    assert tensor.kind == "tensor"
    assert tensor.value == [1, 2, 3]
    assert tensor.dtype == "bf16"
    assert tensor.device == "xpu:0"


def test_to_infinicore_tensor_leaves_existing_device_tensors_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_xpu.model_executor.conversion.get_infinicore",
        lambda: FakeInfinicoreModule(),
    )
    tensor = SimpleNamespace(_underlying="tensor")

    assert to_infinicore_tensor(tensor) is tensor
