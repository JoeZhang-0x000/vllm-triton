"""Minimal vLLM XPU local generation example."""

from __future__ import annotations

import argparse

DEFAULT_PROMPT = "Write one short sentence about large language model inference."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Basic local LLM example")
    parser.add_argument("--model", required=True, help="Model path or Hugging Face identifier")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Plain-text prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling cutoff")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import vllm_xpu

    vllm_xpu.patch_for_vllm_import()
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    llm = LLM(model=args.model, dtype="bfloat16")
    outputs = llm.generate([args.prompt], sampling_params)

    output = outputs[0]
    generated = output.outputs[0].text if hasattr(output, "outputs") else output
    print(generated)


if __name__ == "__main__":
    main()
