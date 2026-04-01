from __future__ import annotations

from pathlib import Path


def test_pyproject_declares_vllm_entrypoints() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.platform_plugins"]' in content
    assert 'xpu = "vllm_xpu.plugins:register_xpu_platform"' in content
    assert '[project.entry-points."vllm.general_plugins"]' in content
    assert (
        'xpu_bootstrap = "vllm_xpu.plugins:register_xpu_general_plugin"'
        in content
    )
