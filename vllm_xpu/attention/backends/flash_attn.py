"""Minimal XPU attention backend compatible with vLLM's backend interfaces."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised when torch is absent.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from vllm_xpu.ops.registry import resolve_operator
from vllm_xpu.runtime.registry import resolve_runtime_adapter

try:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadataBuilder,
        AttentionType,
        CommonAttentionMetadata,
        MultipleOf,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec
except ImportError:  # pragma: no cover - exercised when vllm is absent.
    M = TypeVar("M")

    class AttentionType:
        DECODER = "decoder"
        ENCODER = "encoder"
        ENCODER_ONLY = "encoder_only"
        ENCODER_DECODER = "encoder_decoder"

    class MultipleOf:
        def __init__(self, base: int):
            self.base = base

    class AttentionBackend:
        accept_output_buffer: bool = False
        supported_dtypes: ClassVar[list[torch.dtype]] = []
        supported_kv_cache_dtypes: ClassVar[list[str]] = []
        forward_includes_kv_cache_update: bool = True

    class AttentionImpl:
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

    @dataclass
    class CommonAttentionMetadata:
        query_start_loc: torch.Tensor
        query_start_loc_cpu: torch.Tensor
        seq_lens: torch.Tensor
        num_reqs: int
        num_actual_tokens: int
        max_query_len: int
        max_seq_len: int
        block_table_tensor: torch.Tensor
        slot_mapping: torch.Tensor
        causal: bool = True

    class AttentionMetadataBuilder(Generic[M]):
        supports_update_block_table: bool = False

        def __init__(
            self,
            kv_cache_spec: Any,
            layer_names: list[str],
            vllm_config: Any,
            device: Any,
        ) -> None:
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device

    class AttentionSpec:
        block_size: int

    class VllmConfig:
        pass


@dataclass
class XPUAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None


class XPUAttentionMetadataBuilder(AttentionMetadataBuilder[XPUAttentionMetadata]):
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = getattr(kv_cache_spec, "block_size", 16)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> XPUAttentionMetadata:
        del fast_build
        return XPUAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
            use_cascade=common_prefix_len > 0,
            common_prefix_len=common_prefix_len,
        )

    def update_block_table(
        self,
        metadata: XPUAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> XPUAttentionMetadata:
        updated = copy.copy(metadata)
        updated.block_table = blk_table
        updated.slot_mapping = slot_mapping
        return updated


class XPUAttentionImpl(AttentionImpl):
    """Simple inference-only attention implementation."""

    def __init__(
        self,
        num_heads: int = 1,
        head_size: int = 1,
        scale: float = 1.0,
        num_kv_heads: int = 1,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **_: Any,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.sinks = sinks

    def forward(
        self,
        layer: Any,
        query: Any | None = None,
        key: Any | None = None,
        value: Any | None = None,
        kv_cache: Any | None = None,
        attn_metadata: XPUAttentionMetadata | None = None,
        output: Any | None = None,
        output_scale: Any | None = None,
        output_block_scale: Any | None = None,
        runtime_name: str | None = None,
    ) -> Any:
        if runtime_name is not None and kv_cache is None and attn_metadata is None:
            resolved = resolve_operator("attention", runtime_name=runtime_name)
            return resolved.implementation.forward(layer, query, key)
        del layer, output_scale, output_block_scale
        _require_torch()

        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output.zero_()

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output.zero_()

        if self.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            assert key is not None and value is not None
            return self._forward_direct(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                causal=attn_metadata.causal,
            )

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        provider_output = self._try_provider(
            query[:num_actual_tokens],
            key[:num_actual_tokens] if key is not None else None,
            value[:num_actual_tokens] if value is not None else None,
            key_cache,
            value_cache,
            attn_metadata,
        )
        if provider_output is not None:
            output[:num_actual_tokens].copy_(provider_output)
            return output

        return self._forward_from_cache(
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            output[:num_actual_tokens],
            attn_metadata,
        )

    def do_kv_cache_update(
        self,
        layer: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        slot_mapping: Any,
    ) -> None:
        del layer
        _require_torch()
        if kv_cache.numel() == 0 or key is None or value is None:
            return

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        block_size = key_cache.shape[-2]
        slots = slot_mapping.flatten()
        num_tokens = min(key.shape[0], slots.shape[0])

        for token_idx in range(num_tokens):
            slot = int(slots[token_idx].item())
            if slot < 0:
                continue
            block_idx = slot // block_size
            block_offset = slot % block_size
            key_cache[block_idx, :, block_offset, :].copy_(key[token_idx])
            value_cache[block_idx, :, block_offset, :].copy_(value[token_idx])

    def _try_provider(
        self,
        query: Any,
        key: Any | None,
        value: Any | None,
        key_cache: Any,
        value_cache: Any,
        attn_metadata: XPUAttentionMetadata,
    ) -> Any | None:
        try:
            runtime_name, _ = resolve_runtime_adapter()
            resolved = resolve_operator("attention", runtime_name=runtime_name)
            history_lens = self._history_lens(attn_metadata)
            output = resolved.implementation.forward(
                query,
                key,
                value,
                k_cache=key_cache,
                v_cache=value_cache,
                block_table=attn_metadata.block_table,
                cache_lens=attn_metadata.seq_lens,
                history_lens=history_lens,
                cu_seqlens_q=attn_metadata.query_start_loc,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
            )
        except Exception:
            return None

        if torch is None or not isinstance(output, torch.Tensor):
            return None
        if output.shape == query.shape:
            return output
        if output.ndim == 2 and output.shape[-1] == self.num_heads * self.head_size:
            return output.view(-1, self.num_heads, self.head_size)
        return None

    def _forward_direct(
        self,
        query: Any,
        key: Any,
        value: Any,
        output: Any,
        attn_metadata: XPUAttentionMetadata,
        *,
        causal: bool,
    ) -> Any:
        starts = attn_metadata.query_start_loc.tolist()
        for req_idx in range(len(starts) - 1):
            start = starts[req_idx]
            end = starts[req_idx + 1]
            if end <= start:
                continue
            output[start:end].copy_(
                self._run_attention(query[start:end], key[start:end], value[start:end], causal=causal)
            )
        return output

    def _forward_from_cache(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        output: Any,
        attn_metadata: XPUAttentionMetadata,
    ) -> Any:
        starts = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        block_table = attn_metadata.block_table
        block_size = key_cache.shape[-2]

        for req_idx in range(len(starts) - 1):
            start = starts[req_idx]
            end = starts[req_idx + 1]
            if end <= start:
                continue

            seq_len = int(seq_lens[req_idx])
            q = query[start:end]
            k_ctx, v_ctx = self._gather_cache_sequence(
                key_cache,
                value_cache,
                block_table[req_idx],
                seq_len,
                block_size,
            )
            output[start:end].copy_(self._run_attention(q, k_ctx, v_ctx, causal=True))

        return output

    def _gather_cache_sequence(
        self,
        key_cache: Any,
        value_cache: Any,
        blocks: Any,
        seq_len: int,
        block_size: int,
    ) -> tuple[Any, Any]:
        key_tokens = []
        value_tokens = []
        for token_pos in range(seq_len):
            block_idx = int(blocks[token_pos // block_size].item())
            block_offset = token_pos % block_size
            key_tokens.append(key_cache[block_idx, :, block_offset, :])
            value_tokens.append(value_cache[block_idx, :, block_offset, :])
        return torch.stack(key_tokens, dim=0), torch.stack(value_tokens, dim=0)

    def _run_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        *,
        causal: bool,
    ) -> Any:
        _require_torch()
        key = self._expand_kv_heads(key)
        value = self._expand_kv_heads(value)

        q = query.transpose(0, 1).unsqueeze(0)
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        if causal:
            q_len = query.shape[0]
            k_len = key.shape[0]
            q_positions = torch.arange(
                k_len - q_len, k_len, device=query.device, dtype=torch.int64
            )
            k_positions = torch.arange(k_len, device=query.device, dtype=torch.int64)
            invalid = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
            attn_mask = torch.zeros(
                (q_len, k_len), device=query.device, dtype=torch.float32
            )
            attn_mask.masked_fill_(invalid, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )
        return attended.squeeze(0).transpose(0, 1).contiguous()

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        _require_torch()
        if self.num_heads == self.num_kv_heads:
            return x
        repeats = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeats, dim=1)

    def _split_kv_cache(
        self,
        kv_cache: Any,
    ) -> tuple[Any, Any]:
        if kv_cache.dim() >= 2 and kv_cache.shape[0] == 2:
            return kv_cache[0], kv_cache[1]
        if kv_cache.dim() >= 2 and kv_cache.shape[1] == 2:
            return kv_cache.unbind(1)
        raise ValueError(f"Unsupported kv_cache shape: {tuple(kv_cache.shape)}")

    def _history_lens(self, attn_metadata: XPUAttentionMetadata) -> Any:
        query_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        return attn_metadata.seq_lens - query_lens


class XPUAttentionBackend(AttentionBackend):
    """Simple attention backend for the XPU foundation path."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[Any]] = (
        []
        if torch is None
        else [torch.float16, torch.bfloat16, torch.float32]
    )
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "XPU_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[XPUAttentionImpl]:
        return XPUAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[XPUAttentionMetadataBuilder]:
        return XPUAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (2, num_blocks, num_kv_heads, block_size, head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def use_cascade_attention(*args: Any, **kwargs: Any) -> bool:
        return False


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("torch is required to execute XPU attention backend")
