from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm_xpu.attention.backends.flash_attn import (
    XPUAttentionBackend,
    XPUAttentionImpl,
    XPUAttentionMetadataBuilder,
)


def test_metadata_builder_copies_common_attention_fields() -> None:
    builder = XPUAttentionMetadataBuilder(
        kv_cache_spec=SimpleNamespace(block_size=4),
        layer_names=["layer0"],
        vllm_config=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    common = SimpleNamespace(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        max_seq_len=3,
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0], [0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
        causal=True,
    )

    metadata = builder.build(0, common)

    assert metadata.num_actual_tokens == 3
    assert metadata.max_query_len == 2
    assert metadata.block_table.shape == (2, 1)
    assert metadata.slot_mapping.tolist() == [0, 1, 2]


def test_simple_attention_impl_updates_kv_cache_and_runs_decode() -> None:
    impl = XPUAttentionImpl(
        num_heads=2,
        head_size=4,
        scale=0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 2, 4)
    value = torch.randn(2, 2, 4)
    kv_cache = torch.zeros(2, 1, 2, 4, 4)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)

    impl.do_kv_cache_update(None, key, value, kv_cache, slot_mapping)

    metadata = SimpleNamespace(
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        causal=True,
    )
    output = torch.empty_like(query)

    result = impl.forward(None, query, key, value, kv_cache, metadata, output=output)

    assert result.shape == query.shape
    assert torch.count_nonzero(kv_cache) > 0


def test_backend_exposes_minimal_vllm_contract() -> None:
    assert XPUAttentionBackend.get_name() == "XPU_FLASH_ATTN"
    assert XPUAttentionBackend.get_impl_cls() is XPUAttentionImpl
    assert XPUAttentionBackend.get_builder_cls() is XPUAttentionMetadataBuilder
    assert XPUAttentionBackend.get_kv_cache_shape(2, 4, 3, 8) == (2, 2, 3, 4, 8)
