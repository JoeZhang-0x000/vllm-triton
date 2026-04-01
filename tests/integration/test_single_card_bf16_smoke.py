from __future__ import annotations

import importlib
import sys
import types

from tests.fixtures.runtime_adapters import MockRuntimeAdapter
from vllm_xpu import bootstrap
from vllm_xpu.plugins import register_xpu_platform
from vllm_xpu.runtime.registry import clear_runtime_adapters, register_runtime_adapter


def setup_function() -> None:
    bootstrap._reset_for_tests()
    clear_runtime_adapters()
    sys.modules.pop("vllm", None)


def test_python_only_bootstrap_smoke_path() -> None:
    register_runtime_adapter("mock", MockRuntimeAdapter())

    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.platform = register_xpu_platform()

        def generate(self, prompts, sampling_params):
            return {
                "prompts": prompts,
                "sampling": sampling_params.kwargs,
                "platform": self.platform,
            }

    fake_vllm.LLM = LLM
    fake_vllm.SamplingParams = SamplingParams
    sys.modules["vllm"] = fake_vllm

    module = importlib.import_module("vllm_xpu")
    importlib.reload(module)

    from vllm import LLM, SamplingParams  # type: ignore

    llm = LLM(model="mock-model", dtype="bfloat16")
    sampling_params = SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0, top_k=-1)
    output = llm.generate(["hello"], sampling_params)

    assert bootstrap.is_initialized() is True
    assert output["platform"] == "vllm_xpu.platforms.xpu.XPUPlatform"
    assert output["prompts"] == ["hello"]
    assert output["sampling"] == {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }
