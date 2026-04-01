"""Debug helper for host-side vLLM bootstrap on XPU environments."""

from __future__ import annotations


def main() -> None:
    import vllm_xpu

    applied = vllm_xpu.patch_for_vllm_import()
    print(f"host compat patches: {', '.join(applied) if applied else 'none'}")

    import vllm  # noqa: F401

    print("vllm import: ok")

    from vllm.plugins import load_general_plugins

    load_general_plugins()
    print("general plugins: ok")


if __name__ == "__main__":
    main()
