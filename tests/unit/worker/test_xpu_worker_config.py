from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_xpu.worker.worker import (
    DEFAULT_XPU_WORKER_CLS,
    UnsupportedXPUConfiguration,
    validate_xpu_config,
)


def make_config(
    *,
    dtype="bfloat16",
    quantization=None,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    speculative_config=None,
    worker_cls="auto",
    block_size=None,
    distributed_executor_backend=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization=quantization, dtype=dtype),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            worker_cls=worker_cls,
            distributed_executor_backend=distributed_executor_backend,
        ),
        cache_config=SimpleNamespace(block_size=block_size),
        speculative_config=speculative_config,
    )


def test_validate_xpu_config_sets_worker_cls_and_default_block_size() -> None:
    config = make_config()

    validate_xpu_config(config)

    assert config.parallel_config.worker_cls == DEFAULT_XPU_WORKER_CLS
    assert config.cache_config.block_size == 16


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantization": "awq"}, "quantization"),
        ({"dtype": "float16"}, "bfloat16"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"data_parallel_size": 2}, "data_parallel_size"),
        ({"distributed_executor_backend": "mp"}, "distributed_executor_backend"),
        ({"speculative_config": object()}, "speculative"),
    ],
)
def test_validate_xpu_config_rejects_out_of_scope_features(kwargs, message) -> None:
    config = make_config(**kwargs)

    with pytest.raises(UnsupportedXPUConfiguration, match=message):
        validate_xpu_config(config)
