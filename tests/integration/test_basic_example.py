from __future__ import annotations

import sys
import types

from examples import basic


class FakeOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [types.SimpleNamespace(text=text)]


def test_basic_example_uses_single_prompt_and_sampling_params(monkeypatch, capsys) -> None:
    fake_vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling_params):
            assert prompts == ["hello xpu"]
            assert sampling_params.kwargs == {
                "max_tokens": 16,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20,
            }
            return [FakeOutput("generated text")]

    fake_vllm.LLM = LLM
    fake_vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basic.py",
            "--model",
            "mock-model",
            "--prompt",
            "hello xpu",
            "--max-tokens",
            "16",
            "--temperature",
            "0.3",
            "--top-p",
            "0.9",
            "--top-k",
            "20",
        ],
    )

    basic.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "generated text"
