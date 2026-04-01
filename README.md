# vllm-xpu Foundation

这个仓库实现一个尽量简洁的 out-of-tree `vllm_xpu` 插件基础框架。

目标不是立刻支持所有设备，而是先把一条最小、稳定、可扩展的路径搭起来：

- 用户只安装 `vllm-cpu`
- Python 里先 `import vllm_xpu`
- 后续再通过 runtime adapter 与 operator provider 接入 GPU / 类 GPU 设备

## 当前状态

当前已经完成：

- Python 包骨架与 vLLM plugin entry points
- package bootstrap
- runtime adapter registry
- operator provider registry
- 最小 `XPUPlatform` / `XPUWorker` 桥接层
- 默认 Triton provider
- layer / attention 分发封装

当前还没有完成：

- 真实 vLLM 运行时端到端集成
- 真正的 Triton kernel 实现
- 单卡 BF16 模型推理打通

## 第一阶段支持矩阵

第一阶段明确只针对下面这组范围设计：

- vLLM V1
- 单卡
- BF16
- Python-only 启用方式
- Llama/Qwen-like decoder-only 路径

第一阶段明确不支持：

- `vllm serve` / CLI 兼容
- 多卡 / 分布式
- MoE
- 量化

## 用户使用方式

当前仓库的用户侧目标路径如下：

```python
import vllm_xpu
from vllm import LLM

llm = LLM(model="your-model", dtype="bfloat16")
```

这条路径的前提是：

- 你已经安装了 `vllm`
- 某个设备后端已经通过 `runtime adapter` 注册了可用 runtime
- 需要的算子组要么由设备 override 提供，要么能回退到默认 Triton provider

## 扩展作者入口

后续设备后端需要接两个点：

- runtime adapter
- operator provider

runtime adapter 负责设备控制语义，operator provider 负责按算子组替换实现。

详细说明见 [xpu-extension-points.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/architecture/xpu-extension-points.md)。
