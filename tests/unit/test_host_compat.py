from __future__ import annotations

import sys
import types

from vllm_xpu_host_compat import patch_for_vllm_import


def setup_function() -> None:
    sys.modules.pop("sitecustomize", None)


def test_patch_for_vllm_import_installs_missing_torch_shims(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")
    fake_torch._inductor = None
    convert_frame = types.ModuleType("torch._dynamo.convert_frame")
    dynamo = types.ModuleType("torch._dynamo")
    dynamo.convert_frame = convert_frame

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
    monkeypatch.setitem(sys.modules, "torch._dynamo.convert_frame", convert_frame)
    monkeypatch.delitem(sys.modules, "torch._inductor", raising=False)

    applied = patch_for_vllm_import()

    assert "torch._inductor" in applied
    assert "torch._dynamo.convert_frame.GraphCaptureOutput" in applied
    assert hasattr(fake_torch._inductor, "config")
    assert fake_torch._inductor.config.compile_threads == 1
    assert hasattr(convert_frame, "GraphCaptureOutput")
    assert hasattr(convert_frame.GraphCaptureOutput, "get_runtime_env")


def test_patch_for_vllm_import_is_safe_without_torch(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "cpuinfo", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "torch._dynamo", raising=False)
    monkeypatch.delitem(sys.modules, "torch._dynamo.convert_frame", raising=False)
    monkeypatch.delitem(sys.modules, "torch._inductor", raising=False)

    applied = patch_for_vllm_import()

    assert "cpuinfo.get_cpu_info" in applied
    assert sys.modules["cpuinfo"].get_cpu_info()["count"] is not None


def test_sitecustomize_imports_without_loading_vllm_xpu_package(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "vllm_xpu", raising=False)

    import sitecustomize  # noqa: F401

    assert "vllm_xpu" not in sys.modules
