---
title: feat: 以 Infinicore 跑通 qwen3 dense 最小闭环
type: feat
status: active
date: 2026-04-01
origin: docs/brainstorms/vllm-xpu-requirements.md
---

# feat: 以 Infinicore 跑通 qwen3 dense 最小闭环

## 概述

把 `vllm_xpu` 第一阶段从“默认 Triton provider + 通用 runtime adapter 骨架”重规划为“默认 `Infinicore` 执行基座 + `qwen3 dense` 最小真实闭环”。

这份计划保留统一的 `xpu` 产品入口与 Python-only bootstrap 方式，但不再把 Triton fallback 当成默认执行路径。第一阶段的成功标准改为：

- 在非 PyTorch-native XPU 上，通过 `import vllm_xpu` + `vllm.LLM(...)` 跑通 `qwen3 dense` 架构路径
- 用户侧提供一个仿照 vLLM `basic.py` 的最小本地脚本，支持 `--model`
- 首版只覆盖 plain text prompt、non-streaming、single-prompt 文本生成

这份计划实际取代 [2026-04-01-001-feat-vllm-xpu-foundation-plan.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/plans/2026-04-01-001-feat-vllm-xpu-foundation-plan.md) 里的“默认 Triton provider”方向，但不否定其中已经落地的 plugin / registry / platform 基础骨架。新的工作重点是把这些骨架重新对齐到 `Infinicore-first` 路径。

## 问题背景

origin requirements 已经把产品目标重新收敛为：`vllm_xpu` 不是先做“任意后端通用框架”，而是先做一条真实可跑的 `Infinicore` 路径（见 origin: `docs/brainstorms/vllm-xpu-requirements.md`）。

当前仓库里的基础骨架已经存在：

- `vllm_xpu.plugins` / `vllm_xpu.bootstrap` 负责 vLLM plugin bootstrap
- `vllm_xpu.platforms.xpu.XPUPlatform`、`vllm_xpu.worker.worker.XPUWorker` 负责最小平台/worker 桥接
- `vllm_xpu.runtime.registry` 与 `vllm_xpu.ops.registry` 已建立运行时与算子 provider 的解析机制

但当前默认路径仍然指向 Triton 占位实现，且 worker/model runner 仍带有明显的 `torch.device(...)` 预设，因此无法支撑“不支持 torch 原生设备能力的 XPU + Infinicore”这一真实目标。

与此同时，vendored `Infinicore` 已经具备本阶段关键能力：

- 设备 / dtype / tensor 抽象：`thrid/infinicore/python/infinicore/device.py`、`thrid/infinicore/python/infinicore/dtype.py`、`thrid/infinicore/python/infinicore/tensor.py`
- 基础层能力：`linear`、`RMSNorm`、`RoPE`、`SwiGLU`
- 注意力与 KV cache：`paged_attention`、`paged_attention_prefill`、`mha_kvcache`
- 采样基础件：`random_sample`、`topk`

因此第一阶段的正确问题不再是“如何给 registry 塞一个更像真的 provider”，而是“如何把现有 vLLM XPU 骨架重新对齐到一个默认 `Infinicore` 执行世界”。

## 本地调研摘要

### 代码与结构

- Python 包为单一 Python repo，核心实现位于 `vllm_xpu/`
- 已有计划文档：`docs/plans/2026-04-01-001-feat-vllm-xpu-foundation-plan.md`
- 当前示例脚本为 `examples/offline_inference.py`，接口与目标 `basic.py` 风格不一致
- 当前内建默认 provider 注册逻辑位于 `vllm_xpu/bootstrap.py` 和 `vllm_xpu/ops/registry.py`

### 现有模式

- plugin bootstrap 模式已经稳定，可继续沿用：`vllm_xpu/plugins.py`
- runtime/provider registry 语义已经稳定，可保留为控制面，但不应继续绑死到 Triton 默认值
- `XPUWorker.init_device()` 当前直接尝试 `torch.device(...)`，这是本次重规划的首要收缩点：`vllm_xpu/worker/worker.py`

### Institutional Learnings

- 仓库中不存在 `docs/solutions/`，没有可继承的 institutional learnings

### 外部调研决策

你的代码库已经具备足够强的本地模式和 vendored `Infinicore` 能力面，这份计划不依赖外部调研。最大约束来自当前 `vllm_xpu` 桥接结构与本地 `Infinicore` 实际接口，而不是框架最佳实践缺失。

## 需求映射

- R1. 保持单一 `xpu` 产品入口
- R2. 只支持单卡 BF16
- R3. 保持 Python-only：`import vllm_xpu` 后继续使用 `vllm.LLM(...)`
- R4. 默认执行基座切换为 `Infinicore`
- R5. 用 `Infinicore` 覆盖 `qwen3 dense` 最小执行闭环
- R6. 收紧 `torch` / `Infinicore` 边界，允许必要桥接，但不再让 `torch` 成为内部真相
- R7. 优先复用 vLLM plugin / platform seam
- R8. 成功标准是非 PyTorch-native XPU 上真实跑通 `qwen3 dense`
- R8.1. 提供一个 `basic.py` 风格的最小示例脚本，支持 `--model`
- R8.2. 面向 `qwen3 dense` 架构路径，而非特定 checkpoint
- R8.3. 跑通优先于边界纯度
- R8.4. 贴近常规文本生成，但只限文本主路径
- R8.5. 只支持 plain text prompt
- R8.6. 不做 streaming
- R8.7. 不做 batching，只保证 single-prompt
- R9. 不做量化、多卡、通信、MoE
- R10. 不为未来多默认执行基座预埋抽象

## 规划阶段决策

- **保留 plugin / platform / registry 骨架，但替换内建默认值。**
  - 理由：当前骨架已经承接了 `import vllm_xpu` 与 vLLM plugin seam，推倒重来只会放大风险。

- **第一阶段默认只内建一个 `Infinicore` runtime adapter。**
  - 理由：requirements 已明确首版不要求并列默认基座，`runtime adapter` 抽象可保留，但默认值必须收敛。

- **保留 operator provider registry，但把默认 provider 换成 `Infinicore` provider。**
  - 理由：现有 registry 仍适合表达“默认实现”与“后续 override”；真正该放弃的是 Triton-default 设定，而不是 registry 本身。

- **采样不纳入首版 device provider 覆盖面，继续优先复用 vLLM 宿主侧采样。**
  - 理由：`Infinicore` 虽然已有 `random_sample` / `topk`，但首版目标是先跑通 `qwen3 dense` 执行闭环。把采样策略完全下沉到 XPU 会明显扩大范围。第一阶段只要求 logits 能稳定回到宿主侧，复用现有 non-streaming single-prompt 采样链路即可。

- **首版“常规文本生成体验”明确收敛为 `SamplingParams` 的一个最小子集。**
  - 包含：`max_tokens`、`temperature`、`top_p`、`top_k`、`eos_token_id` / `stop_token_ids`
  - 排除：`streaming`、`messages/chat template`、`batching`、`beam search`、`logprobs`、`prompt_logprobs`、presence/frequency penalty、多模态输入、工具调用

- **第一阶段 `qwen3 dense` 的最小公共算子面定义如下。**
  - `embedding`
  - `linear` / `lm_head`
  - `rms_norm`
  - `swiglu` / `silu_and_mul`
  - `rotary`
  - `paged attention prefill`
  - `paged attention decode + KV cache`
  - logits egress 到宿主采样边界

## 高层技术设计

```text
user
  -> import vllm_xpu
  -> vllm plugin bootstrap
  -> XPUPlatform / XPUWorker
  -> Infinicore runtime adapter
  -> Infinicore default operator provider
  -> qwen3 dense forward path on Infinicore tensors
  -> logits bridged back to vLLM host sampling path
  -> non-streaming single prompt text output
```

边界定义如下：

```text
host / vllm side:
  config
  plugin discovery
  Python LLM entrypoint
  single-prompt non-streaming sampling policy

infinicore side:
  device selection
  tensor allocation / movement
  qwen3 dense forward compute
  kv-cache read/write
  attention prefill + decode
```

## 实施单元

- [ ] **Unit 1: 内建默认 runtime / provider 从 Triton 切换到 Infinicore**

**目标：** 保留现有 bootstrap 与 registry 结构，但把“内建默认值”从 Triton 路线整体切换到 `Infinicore`。

**对应需求：** R1, R3, R4, R7, R10

**依赖：** origin requirements

**文件：**
- 新建：`vllm_xpu/runtime/infinicore.py`
- 修改：`vllm_xpu/runtime/defaults.py`
- 修改：`vllm_xpu/bootstrap.py`
- 修改：`vllm_xpu/plugins.py`
- 修改：`vllm_xpu/platforms/xpu.py`
- 测试：`tests/unit/runtime/test_runtime_registry.py`
- 新建测试：`tests/unit/runtime/test_infinicore_runtime_adapter.py`
- 修改测试：`tests/unit/test_python_bootstrap.py`

**方案：**
- 在 `vllm_xpu.runtime.infinicore` 中实现内建 `InfinicoreRuntimeAdapter`，把 `device_type`、`dispatch_key`、`set_device`、同步、显存查询等语义统一代理到 `infinicore.context` / `infinicore.device`。
- `register_default_runtime_adapters()` 不再为空实现，而是在 `Infinicore` 可导入时注册这一默认 runtime。
- `bootstrap.initialize()` 不再注册 Triton default provider，而是注册 `Infinicore` default provider。
- `XPUPlatform.sync_runtime_metadata()` 继续通过 active runtime 投影平台元数据，但默认命中 `InfinicoreRuntimeAdapter`。

**参考模式：**
- `vllm_xpu/runtime/registry.py`
- `vllm_xpu/plugins.py`
- `thrid/infinicore/python/infinicore/__init__.py`

**测试场景：**
- `import vllm_xpu` 时，在有 `Infinicore` 的环境中默认能注册一个可用 runtime。
- 无可用 runtime 时 plugin 仍应安全返回 `None`。
- runtime metadata 必须来自 `InfinicoreRuntimeAdapter`，而不是硬编码 `"xpu"` 或 `"torch"`。

**完成判定：**
- 包内 bootstrap 的默认执行基座已从 Triton 语义切换为 `Infinicore`。

- [ ] **Unit 2: 建立 Infinicore default provider，并把最小算子面补齐到 qwen3 dense 所需集合**

**目标：** 用 `Infinicore` 实现默认 provider，替换现有 Triton 占位实现，并补齐 `qwen3 dense` 最小前向链路所需算子。

**对应需求：** R4, R5, R8.2

**依赖：** Unit 1

**文件：**
- 修改：`vllm_xpu/ops/groups.py`
- 新建：`vllm_xpu/ops/infinicore/__init__.py`
- 新建：`vllm_xpu/ops/infinicore/attention.py`
- 新建：`vllm_xpu/ops/infinicore/linear.py`
- 新建：`vllm_xpu/ops/infinicore/norm_act.py`
- 新建：`vllm_xpu/ops/infinicore/rotary.py`
- 新建：`vllm_xpu/ops/infinicore/embedding.py`
- 新建：`vllm_xpu/ops/providers/infinicore_provider.py`
- 修改：`vllm_xpu/ops/registry.py`
- 新建：`vllm_xpu/model_executor/layers/embedding.py`
- 修改：`vllm_xpu/model_executor/layers/__init__.py`
- 修改：`vllm_xpu/model_executor/layers/linear.py`
- 修改：`vllm_xpu/model_executor/layers/layernorm.py`
- 修改：`vllm_xpu/model_executor/layers/activation.py`
- 修改：`vllm_xpu/model_executor/layers/rotary_embedding.py`
- 修改：`vllm_xpu/attention/backends/flash_attn.py`
- 测试：`tests/unit/ops/test_operator_registry.py`
- 新建测试：`tests/unit/model_executor/layers/test_embedding_dispatch.py`
- 修改测试：`tests/unit/model_executor/layers/test_linear_dispatch.py`
- 修改测试：`tests/unit/model_executor/layers/test_norm_act_dispatch.py`
- 修改测试：`tests/unit/model_executor/layers/test_rotary_dispatch.py`
- 修改测试：`tests/unit/attention/test_flash_attn_backend.py`
- 新建测试：`tests/unit/ops/providers/test_infinicore_provider.py`

**方案：**
- 默认 provider 改为 `Infinicore`，不再把 Triton provider 注册为 package default。
- 把现有四组算子扩成首版真正需要的五组：`attention`、`linear`、`norm_act`、`rotary`、`embedding`。
- `attention` 组内部同时承接 prefill / decode / paged KV cache 逻辑，避免现在就额外引入 `cache` group。
- `norm_act` 组映射到 `rms_norm` 与 `silu_and_mul` / `swiglu`。
- `sampling` 不进入 provider registry；首版保持在宿主边界。

**方向性设计：**

```text
embedding -> infinicore.nn.functional.embedding
linear/lm_head -> infinicore.nn.functional.linear
norm_act -> infinicore.nn.functional.rms_norm / swiglu or silu_and_mul
rotary -> infinicore.nn.modules.rope or functional rope
attention -> infinicore.paged_attention_prefill / paged_attention / mha_kvcache
```

**参考模式：**
- `thrid/infinicore/python/infinicore/nn/functional/linear.py`
- `thrid/infinicore/python/infinicore/nn/functional/rms_norm.py`
- `thrid/infinicore/python/infinicore/nn/functional/swiglu.py`
- `thrid/infinicore/python/infinicore/nn/modules/rope.py`
- `thrid/infinicore/python/infinicore/ops/paged_attention.py`
- `thrid/infinicore/python/infinicore/ops/paged_attention_prefill.py`

**测试场景：**
- default provider 解析默认命中 `Infinicore`。
- `embedding`、`linear`、`norm_act`、`rotary`、`attention` 都能独立通过 dispatch helper 被解析。
- partial override 语义仍然成立：后续如果只 override `attention`，其余组仍落回默认 `Infinicore` provider。

**完成判定：**
- 默认 provider 已能覆盖 `qwen3 dense` 最小前向公共算子面，而非占位对象。

- [ ] **Unit 3: 收缩 worker / model runner 的 torch 预设，把主执行路径切到 Infinicore**

**目标：** 让 `XPUWorker` 与 model-runner bridge 不再把 `torch.device` 当成默认真相，把主执行路径切到 `Infinicore` tensor/device 语义，同时允许首版保留必要的宿主桥接。

**对应需求：** R4, R6, R8.3

**依赖：** Unit 1, Unit 2

**文件：**
- 修改：`vllm_xpu/worker/worker.py`
- 修改：`vllm_xpu/worker/model_runner.py`
- 新建：`vllm_xpu/model_executor/conversion.py`
- 新建：`vllm_xpu/model_executor/logits_bridge.py`
- 修改：`vllm_xpu/worker/memory.py`
- 测试：`tests/unit/worker/test_runtime_adapter_bridge.py`
- 修改测试：`tests/unit/worker/test_xpu_worker_config.py`
- 新建测试：`tests/unit/model_executor/test_infinicore_conversion.py`
- 新建测试：`tests/unit/model_executor/test_logits_bridge.py`

**方案：**
- `XPUWorker.init_device()` 不再优先构造 `torch.device(...)`；改为通过 runtime adapter 返回或构造 `infinicore.device(...)`，必要时只在 vLLM 兼容点保留字符串/轻量对象。
- 在 `vllm_xpu.model_executor.conversion` 中集中定义：
  - host 输入到 `Infinicore` tensor 的转换
  - 权重从宿主张量到 `Infinicore` tensor 的拥有权策略
  - `Infinicore` logits 回宿主采样边界的导出方式
- `logits_bridge` 只负责把最后一步 logits 交还给宿主采样，不在这里复制一套采样策略。
- `worker.memory` 继续通过 runtime adapter 取显存信息，避免引入任何 `torch.cuda` / `torch.xpu` 假设。

**关键决定：**
- 第一阶段允许 logits 在边界回到宿主侧，以换取更小的 device-side 实现面。
- 如果 `Infinicore` / 宿主张量之间的零拷贝不可保证，首版优先选择语义正确、生命周期清晰的拷贝路径。

**测试场景：**
- worker 初始化不需要真实 `torch.device` 即可成功。
- 输入张量、权重、logits 的桥接逻辑具备明确 ownership，不依赖隐式引用存活。
- 在 fake runtime + fake provider 条件下，model runner 仍能解析并路由到 `Infinicore` default path。

**完成判定：**
- 主执行路径已经由 `Infinicore` 承担，`torch` 不再是 worker/model runner 的默认设备语义。

- [ ] **Unit 4: 定义首版用户可见生成能力子集，并把示例脚本对齐到 basic.py 风格**

**目标：** 固化首版验收入口，提供一个最小、可复现、对用户诚实的本地脚本与文档。

**对应需求：** R3, R8, R8.1, R8.4, R8.5, R8.6, R8.7

**依赖：** Unit 3

**文件：**
- 新建：`examples/basic.py`
- 修改：`examples/offline_inference.py`
- 修改：`README.md`
- 修改：`docs/brainstorms/vllm-xpu-requirements.md`
- 新建：`tests/integration/test_basic_example.py`
- 修改：`tests/integration/test_single_card_bf16_smoke.py`

**方案：**
- 新增 `examples/basic.py`，接口对齐 vLLM `basic.py` 风格，最小支持：
  - `--model`
  - 一个固定 plain text prompt
  - `SamplingParams(max_tokens, temperature, top_p, top_k)`
- `offline_inference.py` 保留为兼容入口或 thin wrapper，但 README 的主入口改为 `examples/basic.py`。
- README 明确首版支持矩阵：
  - plain text prompt only
  - non-streaming only
  - single prompt only
  - `SamplingParams` 只承诺最小子集

**测试场景：**
- 示例脚本参数解析与默认 prompt 稳定。
- 示例脚本在 fake `vllm` 环境下会构造 `LLM(model=..., dtype="bfloat16")` 并触发正确 bootstrap。
- smoke path 反映单 prompt、non-streaming 首版承诺，而不是旧版多 prompt 示例。

**完成判定：**
- 仓库内存在一条和 requirements 完全一致的用户入口，不再依赖旧的多 prompt demo。

- [ ] **Unit 5: 收敛文档叙事，避免仓库继续把自己描述成 Triton-default 框架**

**目标：** 把架构文档与扩展点文档从 Triton-default 叙事更新为 `Infinicore-first` 叙事，同时诚实标注首版范围。

**对应需求：** R4, R7, R10

**依赖：** Unit 1, Unit 2, Unit 4

**文件：**
- 修改：`README.md`
- 修改：`docs/architecture/xpu-extension-points.md`
- 修改：`docs/plans/2026-04-01-001-feat-vllm-xpu-foundation-plan.md`

**方案：**
- README 去掉“默认 Triton provider”作为卖点的叙事。
- `xpu-extension-points.md` 改为说明：
  - 第一阶段默认内建的是 `Infinicore`
  - runtime/provider registry 仍存在，但首版不承诺多默认基座
  - 后续 override 的语义从 `Infinicore` default path 出发描述
- 旧 plan 文档保留历史价值，但应明确它已被 `Infinicore-first` 路线取代，避免误导后续执行。

**测试场景：**
- 文档中不再存在“默认 Triton fallback 是第一阶段主路径”这类与现状冲突的描述。
- 用户与后端作者都能从文档中读出同一条事实：首版默认路径是 `Infinicore + qwen3 dense + plain text + non-streaming + single prompt`。

**完成判定：**
- 仓库文档与新的技术路线一致，不再一边实现 `Infinicore`、一边自我描述为 Triton foundation。

## 系统影响

- **平台元数据：** `XPUPlatform.device_type` / `dispatch_key` 的来源将从抽象占位值收敛到 `InfinicoreRuntimeAdapter`。
- **执行边界：** `torch` 从 worker/device 默认语义中退出，只保留在宿主兼容与 logits 采样边界。
- **provider 语义：** default provider 从 Triton 变为 `Infinicore`；后续 override 机制保留，但默认回退目标改变。
- **示例与验收：** 用户面示例从“离线多 prompt demo”收敛成 `basic.py` 风格的真实最小路径。

## 风险与依赖

- **vLLM 内部仍可能更深地依赖 torch 张量语义。**
  - 缓解：首版明确允许 logits 回宿主侧采样，把纯度要求控制在“主执行路径”而不是“全路径”。

- **权重 / 激活跨边界转换可能带来生命周期与性能风险。**
  - 缓解：在 `conversion.py` 中集中建模 ownership，首版优先语义正确，不追求零拷贝。

- **当前 operator group 设计比真实 `qwen3 dense` 面更窄。**
  - 缓解：只新增 `embedding` 一个真正缺失的稳定组，不把 sampling 提前硬塞进 provider registry。

- **CI 很难在没有目标设备时证明端到端真实可运行。**
  - 缓解：保持 unit/integration fake-path 覆盖，再用 `examples/basic.py` 作为硬件环境下的手工验收入口。

- **`Infinicore` 的 Python device 字符串与 vLLM 平台期望可能存在映射偏差。**
  - 缓解：runtime adapter 负责 canonicalize 对外 device metadata，而不是把 `Infinicore` 内部命名直接外泄。

## 验证策略

### 自动化

- registry / bootstrap / worker 的单测继续跑在 fake runtime / fake provider 条件下
- default provider dispatch 单测验证 `Infinicore` 成为默认 provider
- 示例脚本 integration test 验证 CLI 参数、固定 prompt、single-prompt/non-streaming 路径

### 硬件验收

- 使用 `examples/basic.py --model <path-or-id>` 在目标 XPU 上运行
- 验证 `import vllm_xpu` 后可成功构造 `LLM`
- 验证可完成一次 non-streaming plain text 生成
- 验证 `temperature` / `top_p` / `top_k` 改动对输出路径无崩溃

## 文档与运维说明

- 文档必须把第一阶段支持矩阵写死，避免“看上去支持任意设备 / 任意生成能力”的误导。
- 后续如果要恢复“多默认执行基座”或重新引入 Triton default path，应单独立新 plan，而不是把复杂度偷渡回本计划。

## 资料与引用

- **Origin document:** [docs/brainstorms/vllm-xpu-requirements.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/brainstorms/vllm-xpu-requirements.md)
- 现有基础计划：[docs/plans/2026-04-01-001-feat-vllm-xpu-foundation-plan.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/plans/2026-04-01-001-feat-vllm-xpu-foundation-plan.md)
- 相关代码：[vllm_xpu/plugins.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/vllm_xpu/plugins.py)
- 相关代码：[vllm_xpu/bootstrap.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/vllm_xpu/bootstrap.py)
- 相关代码：[vllm_xpu/platforms/xpu.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/vllm_xpu/platforms/xpu.py)
- 相关代码：[vllm_xpu/worker/worker.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/vllm_xpu/worker/worker.py)
- 相关代码：[vllm_xpu/ops/registry.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/vllm_xpu/ops/registry.py)
- 相关代码：[thrid/infinicore/python/infinicore/__init__.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/thrid/infinicore/python/infinicore/__init__.py)
- 相关代码：[thrid/infinicore/python/infinicore/ops/paged_attention.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/thrid/infinicore/python/infinicore/ops/paged_attention.py)
- 相关代码：[thrid/infinicore/python/infinicore/ops/paged_attention_prefill.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/thrid/infinicore/python/infinicore/ops/paged_attention_prefill.py)
- 相关代码：[thrid/infinicore/python/infinicore/nn/modules/rope.py](/Users/bytedance/Desktop/nt_workspace/vllm-triton/thrid/infinicore/python/infinicore/nn/modules/rope.py)
