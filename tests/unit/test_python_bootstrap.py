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

