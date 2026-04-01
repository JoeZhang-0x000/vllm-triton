from __future__ import annotations

import importlib

from vllm_xpu import bootstrap


def test_initialize_is_idempotent() -> None:
    bootstrap._reset_for_tests()

    assert bootstrap.is_initialized() is False
    assert bootstrap.initialize() is True
    assert bootstrap.is_initialized() is True
    assert bootstrap.initialize() is False


def test_import_runs_package_bootstrap() -> None:
    bootstrap._reset_for_tests()

    module = importlib.import_module("vllm_xpu")
    importlib.reload(module)

    assert bootstrap.is_initialized() is True


def test_initialize_is_reentrant_safe(monkeypatch) -> None:
    bootstrap._reset_for_tests()

    from vllm_xpu.ops import registry as ops_registry
    from vllm_xpu.runtime import defaults as runtime_defaults
    from vllm_xpu import vllm_registration

    monkeypatch.setattr(runtime_defaults, "register_default_runtime_adapters", lambda: None)
    monkeypatch.setattr(ops_registry, "register_default_operator_providers", lambda: None)

    def nested_register() -> None:
        assert bootstrap.initialize() is False

    monkeypatch.setattr(vllm_registration, "register_with_vllm", nested_register)

    assert bootstrap.initialize() is True
    assert bootstrap.is_initialized() is True
