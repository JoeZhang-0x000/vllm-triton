from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from vllm_xpu import bootstrap
from vllm_xpu import vllm_registration


class _FakeAttentionBackend:
    path = None


def _install_fake_vllm_modules() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}

    def add_module(name: str) -> ModuleType:
        module = ModuleType(name)
        if "." not in name or name.endswith(
            (
                ".v1",
                ".attention",
                ".backends",
                ".model_executor",
                ".layers",
                ".rotary_embedding",
            )
        ):
            module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module
        return module

    add_module("vllm")
    add_module("vllm.v1")
    add_module("vllm.v1.attention")
    add_module("vllm.v1.attention.backends")
    add_module("vllm.model_executor")
    add_module("vllm.model_executor.layers")
    add_module("vllm.model_executor.layers.rotary_embedding")

    registry_mod = add_module("vllm.v1.attention.backends.registry")

    class FakeBackendEnum:
        CUSTOM = _FakeAttentionBackend

    def register_backend(backend, class_path: str | None = None, is_mamba: bool = False):
        del is_mamba
        backend.path = class_path
        return lambda cls: cls

    def get_path() -> str:
        return _FakeAttentionBackend.path

    _FakeAttentionBackend.get_path = staticmethod(get_path)
    registry_mod.AttentionBackendEnum = FakeBackendEnum
    registry_mod.register_backend = register_backend

    custom_op_mod = add_module("vllm.model_executor.custom_op")

    class _Registerable:
        registered_names: list[str] = []

        @classmethod
        def register_oot(cls, _decorated=None, name: str | None = None):
            def decorator(op_cls):
                cls.registered_names.append(name or cls.__name__)
                return op_cls

            if _decorated is None:
                return decorator
            return decorator(_decorated)

    custom_op_mod.CustomOp = _Registerable
    custom_op_mod.PluggableLayer = _Registerable

    activation_mod = add_module("vllm.model_executor.layers.activation")
    activation_mod.SiluAndMul = type("SiluAndMul", (_Registerable,), {})

    layernorm_mod = add_module("vllm.model_executor.layers.layernorm")
    layernorm_mod.RMSNorm = type("RMSNorm", (_Registerable,), {})

    linear_mod = add_module("vllm.model_executor.layers.linear")
    linear_mod.ColumnParallelLinear = type("ColumnParallelLinear", (_Registerable,), {})
    linear_mod.MergedColumnParallelLinear = type(
        "MergedColumnParallelLinear", (_Registerable,), {}
    )
    linear_mod.QKVParallelLinear = type("QKVParallelLinear", (_Registerable,), {})
    linear_mod.RowParallelLinear = type("RowParallelLinear", (_Registerable,), {})

    rotary_mod = add_module("vllm.model_executor.layers.rotary_embedding.base")
    rotary_mod.RotaryEmbedding = type("RotaryEmbedding", (_Registerable,), {})

    vocab_mod = add_module("vllm.model_executor.layers.vocab_parallel_embedding")
    vocab_mod.VocabParallelEmbedding = type(
        "VocabParallelEmbedding", (_Registerable,), {}
    )

    modules["vllm"].v1 = modules["vllm.v1"]
    modules["vllm.v1"].attention = modules["vllm.v1.attention"]
    modules["vllm.v1.attention"].backends = modules["vllm.v1.attention.backends"]
    modules["vllm.v1.attention.backends"].registry = registry_mod
    modules["vllm"].model_executor = modules["vllm.model_executor"]
    modules["vllm.model_executor"].layers = modules["vllm.model_executor.layers"]
    modules["vllm.model_executor.layers"].activation = activation_mod
    modules["vllm.model_executor.layers"].layernorm = layernorm_mod
    modules["vllm.model_executor.layers"].linear = linear_mod
    modules["vllm.model_executor.layers"].rotary_embedding = modules[
        "vllm.model_executor.layers.rotary_embedding"
    ]
    modules["vllm.model_executor.layers.rotary_embedding"].base = rotary_mod
    modules["vllm.model_executor.layers"].vocab_parallel_embedding = vocab_mod

    sys.modules.update(modules)
    return modules


def test_initialize_registers_custom_attention_backend_when_vllm_is_importable() -> None:
    bootstrap._reset_for_tests()
    vllm_registration._reset_for_tests()
    modules = _install_fake_vllm_modules()

    try:
        assert bootstrap.initialize() is True
        assert _FakeAttentionBackend.path == (
            "vllm_xpu.attention.backends.flash_attn.XPUAttentionBackend"
        )
    finally:
        for name in modules:
            sys.modules.pop(name, None)
        sys.modules.pop("vllm_xpu.oot", None)
