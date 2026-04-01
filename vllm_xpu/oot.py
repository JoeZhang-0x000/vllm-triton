"""Out-of-tree vLLM layer overrides for the Infinicore-backed XPU path."""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_xpu.model_executor.layers.activation import apply_silu_mul
from vllm_xpu.model_executor.layers.embedding import apply_embedding
from vllm_xpu.model_executor.layers.layernorm import apply_rms_norm
from vllm_xpu.model_executor.layers.linear import apply_linear
from vllm_xpu.model_executor.layers.rotary_embedding import apply_rotary_embedding


def _supports_unquantized_single_rank(layer: Any) -> bool:
    return (
        getattr(layer, "tp_size", 1) == 1
        and getattr(getattr(layer, "quant_method", None), "__class__", type(None)).__name__
        == "UnquantizedLinearMethod"
    )


def _column_like_forward(layer: Any, input_: Any) -> Any:
    if not _supports_unquantized_single_rank(layer):
        return super(type(layer), layer).forward(input_)

    bias = layer.bias if not layer.skip_bias_add else None
    output = apply_linear(input_, layer.weight, bias)
    if not layer.return_bias:
        return output
    output_bias = layer.bias if layer.skip_bias_add else None
    return output, output_bias


def _row_like_forward(layer: Any, input_: Any) -> Any:
    if not _supports_unquantized_single_rank(layer):
        return super(type(layer), layer).forward(input_)

    bias = None if layer.skip_bias_add else layer.bias
    output = apply_linear(input_, layer.weight, bias)
    if not layer.return_bias:
        return output
    output_bias = layer.bias if layer.skip_bias_add else None
    return output, output_bias


@RMSNorm.register_oot(name="RMSNorm")
class XPURMSNorm(RMSNorm):
    def forward_oot(
        self,
        x: Any,
        residual: Any | None = None,
    ) -> Any:
        if residual is not None:
            x = x + residual
            residual = x
        output = apply_rms_norm(x, self.weight, eps=self.variance_epsilon)
        if residual is None:
            return output
        return output, residual


@SiluAndMul.register_oot(name="SiluAndMul")
class XPUSiluAndMul(SiluAndMul):
    def forward_oot(self, x: Any) -> Any:
        gate_dim = x.shape[-1] // 2
        return apply_silu_mul(x[..., :gate_dim], x[..., gate_dim:])


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class XPURotaryEmbedding(RotaryEmbedding):
    def forward_oot(
        self,
        positions: Any,
        query: Any,
        key: Any | None = None,
    ) -> tuple[Any, Any | None]:
        if key is None:
            return self.forward_native(positions, query, key)
        cos_sin = self._match_cos_sin_cache_dtype(query).index_select(
            0, positions.flatten()
        )
        cos, sin = cos_sin.chunk(2, dim=-1)
        return apply_rotary_embedding(
            query,
            key,
            cos,
            sin,
            position_ids=positions,
        )


@ColumnParallelLinear.register_oot(name="ColumnParallelLinear")
class XPUColumnParallelLinear(ColumnParallelLinear):
    def forward(self, input_: Any) -> Any:
        return _column_like_forward(self, input_)


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class XPUMergedColumnParallelLinear(MergedColumnParallelLinear):
    def forward(self, input_: Any) -> Any:
        return _column_like_forward(self, input_)


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class XPUQKVParallelLinear(QKVParallelLinear):
    def forward(self, input_: Any) -> Any:
        return _column_like_forward(self, input_)


@RowParallelLinear.register_oot(name="RowParallelLinear")
class XPURowParallelLinear(RowParallelLinear):
    def forward(self, input_: Any) -> Any:
        return _row_like_forward(self, input_)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class XPUVocabParallelEmbedding(VocabParallelEmbedding):
    def forward_oot(self, input_: Any) -> Any:
        if getattr(self, "tp_size", 1) != 1:
            return self.forward_native(input_)
        return apply_embedding(input_.long(), self.weight)
