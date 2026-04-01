"""Minimal Python-only bootstrap example for vllm_xpu."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline inference example")
    parser.add_argument("model", help="Model path or Hugging Face identifier")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import vllm_xpu  # noqa: F401
    from vllm import LLM, SamplingParams

    prompts = [
        "The future of AI is",
        "vLLM XPU aims to",
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

    llm = LLM(model=args.model, dtype="bfloat16")
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()

