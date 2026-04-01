# vLLM XPU 扩展点

本文档说明 `vllm_xpu` 第一阶段暴露给后续设备后端的最小扩展面。

## 目标

`vllm_xpu` 当前不把自己定义成“任意默认执行基座都平等存在”的大框架。第一阶段先固定一条真实可跑路径：

- 对外仍然只有统一的 `xpu` 产品入口
- 对内默认执行基座固定为 `Infinicore`
- 后续设备后端仍可以通过 runtime adapter 与 operator provider 做定制覆盖

因此，当前扩展设计只解决两个问题：

- 如何让一个设备声明“我是一种可用 runtime”
- 如何让一个设备只替换自己真正需要替换的算子组

这两个问题分别由 `runtime adapter` 和 `operator provider` 负责。

## 结构概览

`vllm_xpu` 当前有两层注册表：

- `vllm_xpu.runtime.registry`
  - 管理 runtime adapter
  - 负责 active runtime 选择
  - 给 `XPUPlatform`、`XPUWorker`、`XPUModelRunnerBridge` 提供统一设备接口
- `vllm_xpu.ops.registry`
  - 管理 operator provider
  - 负责默认 provider 与 runtime override 的优先级
  - 给 attention backend 和 layer dispatch helpers 提供统一算子解析

解析顺序固定如下：

```text
runtime adapter:
  active runtime -> 第一个 available runtime -> 报错

operator provider:
  runtime override -> 默认 Infinicore provider -> 报错
```

## Runtime Adapter 合同

设备后端应至少提供以下方法：

- `is_available() -> bool`
- `device_type() -> str`
- `dispatch_key() -> str`
- `set_device(device) -> None`
- `synchronize() -> None`
- `empty_cache() -> None`
- `mem_get_info() -> tuple[int, int]`
- `get_device_name(device_id: int = 0) -> str`
- `get_device_total_memory(device_id: int = 0) -> int`
- `get_device_capability(device_id: int = 0) -> object | None`
- `inference_mode() -> ContextManager`

建议做法：

- 如果设备没有某种“能力值”概念，`get_device_capability` 返回 `None`
- `device_type()` 返回的字符串应稳定，因为 `XPUPlatform.sync_runtime_metadata()` 会把它投影到平台元数据上
- `dispatch_key()` 应尽量对应底层运行时的真实 dispatch key；如果没有，返回一个稳定字符串即可

示例：

```python
from contextlib import nullcontext


class MyRuntimeAdapter:
    def is_available(self) -> bool:
        return True

    def device_type(self) -> str:
        return "myxpu"

    def dispatch_key(self) -> str:
        return "MYXPU"

    def set_device(self, device) -> None:
        ...

    def synchronize(self) -> None:
        ...

    def empty_cache(self) -> None:
        ...

    def mem_get_info(self) -> tuple[int, int]:
        return (free_bytes, total_bytes)

    def get_device_name(self, device_id: int = 0) -> str:
        return f"myxpu:{device_id}"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return total_bytes

    def get_device_capability(self, device_id: int = 0):
        return None

    def inference_mode(self):
        return nullcontext()
```

注册方式：

```python
from vllm_xpu.runtime.registry import register_runtime_adapter

register_runtime_adapter("myxpu", MyRuntimeAdapter())
```

## Operator Provider 合同

当前稳定算子组有五个：

- `attention`
- `embedding`
- `linear`
- `norm_act`
- `rotary`

默认 `Infinicore` provider 会一次性提供这五组能力。后续设备后端不需要完整接管，可以只 override 自己关心的算子组。

示例：只替换 `attention`

```python
from vllm_xpu.ops.provider import OperatorProvider
from vllm_xpu.ops.registry import (
    register_operator_provider,
    register_runtime_provider_override,
)


class MyAttentionOps:
    def forward(self, query, key, value, **kwargs):
        ...


provider = OperatorProvider(
    name="myxpu-attention",
    implementations={"attention": MyAttentionOps()},
)

register_operator_provider("myxpu-attention", provider)
register_runtime_provider_override("myxpu", "myxpu-attention")
```

在这个例子里：

- `attention` 走设备自定义实现
- `embedding`、`linear`、`norm_act`、`rotary` 继续走默认 `Infinicore` provider

## 何时应该新增算子组

不要轻易新增算子组。

当前建议只有在以下条件同时成立时才新增：

- 它代表一个稳定的能力面，而不是临时实现细节
- 设备后端很可能会整体替换这一组，而不是只替换其中单个小函数
- 新增之后不会让 provider 解析逻辑碎片化

如果只是某个算子内部出现了实现分歧，优先在现有算子组内部扩展 provider 实现，而不是马上引入新的 group。

## 第一阶段限制

当前扩展点文档只覆盖第一阶段承诺的范围：

- vLLM V1
- 单卡
- BF16
- `qwen3 dense` 架构路径
- plain text prompt
- non-streaming
- single prompt

以下内容当前不属于稳定扩展面：

- CLI / `vllm serve`
- 多卡 / 分布式
- MoE
- 量化
- chat / messages 输入协议
- batching
- 设备专有 memory pool / graph capture / custom C++ kernels

如果后续工作要引入这些内容，应单独扩展 plan，而不是直接在当前抽象上堆。
