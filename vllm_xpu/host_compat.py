"""Host-side compatibility shims applied before importing vLLM."""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import Any


def _ensure_module(module_name: str) -> types.ModuleType:
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name:
        parent = _ensure_module(parent_name)
        if not hasattr(parent, child_name):
            setattr(parent, child_name, module)

    return module


def _ensure_attr(target: Any, attr: str, value: Any) -> bool:
    if hasattr(target, attr):
        return False
    setattr(target, attr, value)
    return True


def _ensure_inductor_config(torch_module: Any, applied: list[str]) -> None:
    inductor = getattr(torch_module, "_inductor", None)
    if inductor is None:
        inductor = _ensure_module("torch._inductor")
        torch_module._inductor = inductor
        applied.append("torch._inductor")

    config = getattr(inductor, "config", None)
    if config is None:
        config = types.SimpleNamespace()
        inductor.config = config
        applied.append("torch._inductor.config")

    if _ensure_attr(config, "compile_threads", 1):
        applied.append("torch._inductor.config.compile_threads")


def _ensure_graph_capture_output(applied: list[str]) -> None:
    try:
        convert_frame = importlib.import_module("torch._dynamo.convert_frame")
    except Exception:
        convert_frame = _ensure_module("torch._dynamo.convert_frame")
        applied.append("torch._dynamo.convert_frame")

    graph_capture_output = getattr(convert_frame, "GraphCaptureOutput", None)
    if graph_capture_output is None:
        graph_capture_output = type(
            "GraphCaptureOutput",
            (),
            {
                # vLLM may monkey patch this method at import time. Returning an
                # empty runtime environment is sufficient for host bootstrap.
                "get_runtime_env": lambda self: {},
            },
        )
        convert_frame.GraphCaptureOutput = graph_capture_output
        applied.append("torch._dynamo.convert_frame.GraphCaptureOutput")

    if _ensure_attr(graph_capture_output, "get_runtime_env", lambda self: {}):
        applied.append("torch._dynamo.convert_frame.GraphCaptureOutput.get_runtime_env")


def patch_for_vllm_import() -> tuple[str, ...]:
    """Apply minimal host shims required before importing vLLM.

    Returns a tuple describing the shims that were installed for diagnostics.
    """

    applied: list[str] = []
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

    try:
        torch_module = importlib.import_module("torch")
    except ImportError:
        return tuple(applied)

    _ensure_inductor_config(torch_module, applied)
    _ensure_graph_capture_output(applied)
    return tuple(applied)
