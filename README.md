# vllm-xpu

这个仓库实现一个 out-of-tree `vllm_xpu` 插件，当前第一阶段默认以 `Infinicore` 作为执行基座，目标是在非 PyTorch-native XPU 上跑通 `qwen3 dense` 架构路径的最小文本生成闭环。

第一阶段刻意收敛到一条很窄、但真实可跑的路径：

- 用户只安装 `vllm-cpu`
- Python 里先 `import vllm_xpu`
- 默认执行路径走 `Infinicore`
- 验收目标是 plain text、non-streaming、single-prompt 生成

## 当前状态

当前已经完成：

- Python 包骨架与 vLLM plugin entry points
- package bootstrap
- runtime adapter registry
- operator provider registry
- 最小 `XPUPlatform` / `XPUWorker` 桥接层
- 默认 `Infinicore` provider
- layer / attention 分发封装
- `basic.py` 风格的最小本地示例入口

当前还没有完成：

- 真实硬件环境下的 `qwen3 dense` 端到端打通
- 更完整的 vLLM 生成能力覆盖
- `torch` 宿主边界的进一步收缩

## 第一阶段支持矩阵

第一阶段明确只针对下面这组范围设计：

- vLLM V1
- 单卡
- BF16
- Python-only 启用方式
- `qwen3 dense` 架构路径
- plain text prompt
- non-streaming
- single prompt

第一阶段明确不支持：

- `vllm serve` / CLI 兼容
- 多卡 / 分布式
- MoE
- 量化
- chat / messages 输入协议
- streaming
- batching
- 多模态与工具调用

## 用户使用方式

最小本地调用示例如下：

```bash
python examples/basic.py --model your-model
```

等价的 Python 入口仍然是：

```python
import vllm_xpu
from vllm import LLM

llm = LLM(model="your-model", dtype="bfloat16")
```

当前主示例脚本支持：

- `--model`
- `--prompt`
- `--max-tokens`
- `--temperature`
- `--top-p`
- `--top-k`

这条路径当前的前提是：

- 你已经安装了 `vllm`
- `Infinicore` 对目标设备可用
- 需要的算子组要么由设备 override 提供，要么能回退到默认 `Infinicore` provider

## 扩展作者入口

后续设备后端需要接两个点：

- runtime adapter
- operator provider

runtime adapter 负责设备控制语义，operator provider 负责按算子组替换实现。

详细说明见 [xpu-extension-points.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/architecture/xpu-extension-points.md)。
