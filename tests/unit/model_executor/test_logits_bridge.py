from __future__ import annotations

from vllm_xpu.model_executor.logits_bridge import export_logits_to_host


class FakeLogits:
    def to_host(self):
        return [1, 2, 3]


def test_export_logits_to_host_prefers_explicit_to_host_method() -> None:
    assert export_logits_to_host(FakeLogits()) == [1, 2, 3]


def test_export_logits_to_host_returns_input_when_no_bridge_is_needed() -> None:
    logits = {"logits": [0.1, 0.2]}

    assert export_logits_to_host(logits) is logits
