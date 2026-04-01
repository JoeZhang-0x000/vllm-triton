---
title: feat: 收敛 vLLM 宿主 torch 依赖并打通 MLU bring-up
type: feat
status: proposed
date: 2026-04-01
origin: docs/brainstorms/vllm-xpu-requirements.md
related:
  - docs/plans/2026-04-01-002-feat-infinicore-qwen3-dense-plan.md
---

# feat: 收敛 vLLM 宿主 torch 依赖并打通 MLU bring-up

## 概述

当前 `vllm_xpu` 已经把执行层方向收敛为 `Infinicore-first`，并通过 OOT layer override、custom attention backend、runtime/provider registry 接上了上游 `vllm` 的真实接缝。但在真实 MLU bring-up 时，第一批失败并不发生在算子执行，而是发生在 `vllm` 宿主 import/install 侧：

- `VLLM_TARGET_DEVICE=cpu` 安装路径依赖 `torch==2.10.0+cpu`，在目标环境内源不可用
- `VLLM_TARGET_DEVICE=empty` 虽然规避了原生后端编译，但 `vllm` 仍会在 import 阶段加载 `env_override`
- 目标机器上的 `vllm`/`torch` 组合在 `torch._dynamo` 私有 API 上已经出现不兼容，导致 `from vllm import LLM` 之前就失败

因此下一阶段的核心不再是继续扩算子面，而是把问题重新拆成两层：

1. **宿主兼容层**：让 `vllm` 在非标准 `torch` 环境中至少能完成安装、import、插件发现和最小 LLM 入口初始化。
2. **执行层**：在宿主存活的前提下，继续把真实算子执行收敛到 `vllm_xpu + Infinicore`。

这份计划的目标是定义一条“合理、分阶段、可 bring-up”的路径，使 `torch` 在 `vllm` 中只保留宿主壳的作用，而非继续主导执行后端。首选方案不是先改 `../vllm`，而是在 `import vllm` 之前，通过 `vllm_xpu` 提供的 host compatibility monkey patch 先把宿主带起来。

## 问题背景

已有计划 [2026-04-01-002-feat-infinicore-qwen3-dense-plan.md](/Users/bytedance/Desktop/nt_workspace/vllm-triton/docs/plans/2026-04-01-002-feat-infinicore-qwen3-dense-plan.md) 主要聚焦 `vllm_xpu` 仓库内部：默认 runtime/provider 切到 `Infinicore`、补最小算子面、接上上游 OOT seam、提供 `basic.py` 验收入口。

这部分工作已经具备以下基础：

- `vllm_xpu` 已可通过 `vllm.platform_plugins` / `vllm.general_plugins` 完成平台注册与通用 bootstrap。
- `vllm_xpu.vllm_registration` 和 `vllm_xpu.oot` 已经能够把 custom attention backend 和 OOT layer override 注册到上游 `vllm`。
- `vllm_xpu.attention.backends.flash_attn.XPUAttentionBackend` 已有最小 inference-only backend。

但真实机器 bring-up 暴露出一个新的前置阻塞：

- `vllm` 作为宿主，仍然默认假设兼容的 `torch` 安装、私有 `torch._dynamo` API、以及一部分平台级初始化行为。
- 这与需求文档里的 R6 / R8.3 并不矛盾，但说明“`torch` 只留在宿主边界”这句话需要被拆成更具体的工程动作，而不是停留在原则层。

## 本地调研摘要

### 代码与接缝

- `vllm_xpu` 插件入口已就位：`pyproject.toml`、`vllm_xpu/plugins.py`
- 运行时与 provider 默认路径已切到 `Infinicore`：`vllm_xpu/runtime/defaults.py`、`vllm_xpu/ops/registry.py`
- 上游替换接缝已接入：`vllm_xpu/vllm_registration.py`、`vllm_xpu/oot.py`
- 最小 attention backend 已接入：`vllm_xpu/attention/backends/flash_attn.py`

### 上游 vLLM 现状

- `vllm` 支持 `VLLM_TARGET_DEVICE=empty`，并会在该模式下仅安装通用 Python 依赖（见 `../vllm/setup.py`）。
- `vllm.__init__` 会无条件 import `vllm.env_override`，即使走 `empty` 安装路径也无法绕过 import-time 兼容逻辑（见 `../vllm/vllm/__init__.py`）。
- 目标机器的真实报错表明：宿主 `torch` 与 `vllm` 之间已经存在 `torch._dynamo` 私有 API 不兼容，当前 `vllm` 还不够“宿主兼容边界最小化”。
- `vllm` 的 plugin 加载时机在 `import vllm` 之后，因此不能依赖 `vllm.general_plugins` / `vllm.platform_plugins` 去修复 import-time 失败。

### 结构判断

- **本阶段不应该继续扩大 `vllm_xpu` 内部算子范围。**
  第一阻塞点已经从执行算子前移到了宿主 import/install。
- **本阶段优先保持单仓实现。**
  优先在 `vllm-triton` 内提供 pre-import monkey patch、launcher bootstrap 和调试入口；只有当这条路验证失败时，才升级为直接修改 `../vllm`。

### External Research 决策

当前问题高度依赖本地两份代码树的接口和真实报错，不需要外部调研。规划直接基于 `vllm_xpu`、`../vllm` 和 MLU bring-up 日志即可。

## 需求映射

- R3. 用户仍通过 Python 入口 `import vllm_xpu` + `vllm.LLM(...)` 使用系统。
- R4. 默认执行基座仍是 `Infinicore`，本计划不回退到 Triton/PyTorch runtime 主路径。
- R6. `torch` 只能保留在必须的宿主兼容边界；本计划的核心就是把这个边界具象化。
- R7. 优先复用 vLLM plugin / platform seam；本计划不引入第二套模型执行框架。
- R8 / R8.3. 先真实跑通，允许宿主侧保留过渡桥接，但不再扩大 `torch` 在执行层的作用面。

## 关键决策

- **决策 1：先把 `vllm` 定位为“可导入、可初始化、可发现插件的宿主”，而不是先尝试完全去 torch 化。**
  - 理由：当前真实阻塞是 import-time 崩溃。若宿主起不来，执行层优化没有验收路径。

- **决策 2：优先支持 `VLLM_TARGET_DEVICE=empty` 的 bring-up 路线，而不是要求 `vllm` 本体安装任何官方 backend。**
  - 理由：这条路最符合“宿主只做壳、执行走我们后端”的方向，也避开 CPU/XPU/CUDA 官方后端构建依赖。

- **决策 3：优先用 pre-import monkey patch 吸收 import-time torch 私有 API 不兼容，而不是先改 `../vllm`。**
  - 理由：`vllm` plugin 机制时机太晚，只有在 `import vllm` 之前 patch 宿主 `torch`/环境，才有机会绕过 `env_override` 导入失败。

- **决策 4：为 monkey patch 设计两个载体，先显式 bootstrap，再按需要升级到 `sitecustomize`。**
  - 理由：`examples/basic.py` 这类入口可以先显式调用 `patch_for_vllm_import()`；如果后续发现 worker 子进程也需要同样 patch，再收敛到更早的自动注入载体。

- **决策 5：首版不尝试替换 `vllm` 的 `torch.nn.Module` / `Parameter` / 权重加载体系。**
  - 理由：这会把范围迅速膨胀为“另一个宿主框架”。第一阶段只处理 import、platform、backend、最小 tensor bridge。

- **决策 6：采用 characterization-first 执行姿态。**
  - 理由：当前问题强依赖目标机器上的 `vllm` checkout、`torch` 发行版和插件加载顺序。先补 characterization/smoke，再逐层收敛更安全。

## 范围边界

### In Scope

- `../vllm` 的 `empty` 模式安装与 import-time 兼容
- `vllm_xpu` 的平台注册、OOT 注册、最小 backend 接入
- 单卡、BF16、single-prompt、plain-text、non-streaming bring-up
- 真实 MLU 机器上的首次端到端 smoke 验证入口

### Out of Scope

- 去除 `vllm` 宿主全部 `torch` 依赖
- 重写 `qwen3` / `qwen2` 模型实现
- 替换 `nn.Module` / `Parameter` / checkpoint loading 主体系
- 多卡、量化、MoE、streaming、batching

## 高层技术设计

```text
python env
  -> import vllm_xpu.host_compat
  -> patch_for_vllm_import()
  -> import vllm
  -> empty-target host install succeeds
  -> import-time env_override / torch compatibility gates become patch-safe
  -> import vllm_xpu
  -> vllm.general_plugins / vllm.platform_plugins discover xpu plugin
  -> XPUPlatform chooses CUSTOM attention backend + OOT overrides
  -> qwen3/qwen2 model path remains upstream
  -> selected layers / backend dispatch into Infinicore
  -> logits bridge returns to host sampling boundary
```

边界定义：

```text
host side (still torch-based):
  import / config / plugin discovery
  pre-import monkey patch
  model object graph
  parameter loading
  sampling policy

execution side (Infinicore-first):
  runtime adapter
  operator provider
  attention backend
  OOT layer overrides
  kv cache update
```

## 实施单元

- [ ] **Unit 1: 规范宿主基线并引入 pre-import monkey patch 路线**

**目标：** 把“支持哪份 `vllm` 源码、哪种安装方式、哪种宿主 `torch` 约束”明确下来，避免本地规划和目标机器环境继续漂移。

**对应需求：** R3, R6, R8.3

**文件：**
- 修改：`README.md`
- 修改：`docs/architecture/xpu-extension-points.md`
- 新建：`docs/bringup/mlu-host-setup.md`
- 新建：`vllm_xpu/host_compat.py`
- 新建：`tests/unit/test_host_compat.py`
- 新建测试：`tests/integration/test_empty_target_host_assumptions.py`

**方案：**
- 明确第一阶段受支持的宿主安装方式为：`VLLM_TARGET_DEVICE=empty` 的 editable install。
- 新增 `vllm_xpu.host_compat.patch_for_vllm_import()`，集中承载最小宿主兼容逻辑：
  - 预设环境变量
  - 为缺失的 `torch` 私有符号补 stub
  - 对 `_inductor` 配置访问做 feature detection
- 文档中明确要求所有 bring-up 入口在 `import vllm` 之前先执行这层 patch。
- 为 `basic.py` bring-up 补一份宿主检查清单：host patch 已执行、`vllm` 可导入、插件 entrypoint 可发现。

**测试场景：**
- 仓库内 smoke 明确表达 `empty` 路线是受支持 bring-up 路径。
- 文档中给出最小安装/验证步骤，不再误导用户优先走 `cpu` 后端安装。

**完成判定：**
- 团队对“MLU bring-up 依赖什么宿主基线、patch 时机在哪里”有唯一答案，不再靠口头约定。

- [ ] **Unit 2: 用宿主 monkey patch 吸收 import-time 的 torch 私有 API 不兼容**

**目标：** 让 `vllm` 在 `empty` 模式 + 非标准 torch 发行版下，至少可以完成 import 和 `LLM` 入口初始化前的模块加载。

**对应需求：** R3, R6, R7

**文件：**
- 修改：`vllm_xpu/host_compat.py`
- 修改：`examples/basic.py`
- 新建：`examples/debug_host_bootstrap.py`
- 新建测试：`tests/unit/test_host_compat.py`
- 新建测试：`tests/integration/test_empty_target_import.py`

**方案：**
- 审计真实失败所需的最小兼容点，例如 `torch._dynamo.convert_frame.GraphCaptureOutput`。
- 将这些兼容逻辑集中放在 `patch_for_vllm_import()` 内：
  - 能找到原始对象则不处理
  - 缺失时补最小 stub
  - 不支持的 `_inductor` 配置项只做 guarded write
- 保证 `from vllm import LLM` 不会因为图编译相关内部 API 不兼容而整体 import 失败。
- 新增 `debug_host_bootstrap.py`，专门验证“host patch 后可导入 vllm”。

**测试场景：**
- 模拟宿主 `torch` 缺失某些 `torch._dynamo` 私有成员时，执行 host patch 后 `import vllm` 仍然成功。
- `VLLM_TARGET_DEVICE=empty` 场景下，上述 patch 不应成为阻塞。

**完成判定：**
- `import vllm` 在目标机器上不再因 `torch` 私有 API 不兼容而失败，且无需修改 `../vllm`。

- [ ] **Unit 3: 把 `vllm_xpu` 平台选择与 OOT 注册收敛为 empty-host 友好模式**

**目标：** 确保 `vllm` 宿主一旦能 import，插件发现、平台注册、CUSTOM attention backend 注册、OOT layer override 注册都能稳定完成。

**对应需求：** R4, R6, R7

**文件：**
- 修改：`vllm_xpu/bootstrap.py`
- 修改：`vllm_xpu/plugins.py`
- 修改：`vllm_xpu/platforms/xpu.py`
- 修改：`vllm_xpu/vllm_registration.py`
- 修改：`vllm_xpu/oot.py`
- 测试：`tests/unit/test_vllm_registration.py`
- 新建测试：`tests/integration/test_empty_target_plugin_bootstrap.py`

**方案：**
- 把当前 `register_with_vllm()` 从“best-effort”进一步收敛为：
  - 先验证上游 API 存在性
  - 再注册 CUSTOM backend
  - 再加载 OOT override
- 确保 bootstrap 不隐式依赖“用户先 import vllm 再 import vllm_xpu”这类顺序假设，必要时提供单一入口 helper。
- 增加对宿主 `vllm` 版本/接口漂移的显式检测，必要时尽早给出可读错误，而不是静默失败。
- 确保 `XPUPlatform.get_attn_backend_cls()` 在 `empty` host 模式下仍返回我们自己的 backend，不落回内建 CPU/CUDA/XPU 后端选择。

**测试场景：**
- fake `vllm` / minimal `vllm` 环境下，插件初始化不因缺少非关键 API 而崩溃。
- platform 解析明确落到 `vllm_xpu.platforms.xpu.XPUPlatform`。

**完成判定：**
- 宿主与插件的接缝已经稳定，不再依赖“恰好和本地 torch/vllm 组合一致”。

- [ ] **Unit 4: 把 bring-up 路径收敛为最小可诊断闭环**

**目标：** 为真实 MLU 调试建立一个“能跑、能报、能定位”的最小闭环，而不是直接把所有失败都堆给 `examples/basic.py`。

**对应需求：** R5, R8, R8.1, R8.3

**文件：**
- 修改：`examples/basic.py`
- 修改：`examples/offline_inference.py`
- 修改：`vllm_xpu/host_compat.py`
- 修改：`vllm_xpu/worker/worker.py`
- 修改：`vllm_xpu/worker/model_runner.py`
- 修改：`vllm_xpu/model_executor/conversion.py`
- 修改：`vllm_xpu/model_executor/logits_bridge.py`
- 新建：`examples/debug_platform_selection.py`
- 新建测试：`tests/integration/test_basic_example.py`
- 新建测试：`tests/integration/test_empty_target_debug_examples.py`

**方案：**
- `basic.py` 保持用户入口不变，但在失败时输出更具体的阶段信息：host patch、宿主 import、plugin 发现、platform 选择、backend 选择、模型初始化、第一次 generate。
- 新增两个调试脚本：
  - `debug_host_bootstrap.py`：只验证 host patch + `vllm` import + plugin load
  - `debug_platform_selection.py`：验证当前平台是否落到 `XPUPlatform`
- 对 `conversion.py` / `logits_bridge.py` 只做最薄的可诊断增强，不在本阶段引入新的抽象。

**测试场景：**
- 无真实设备时也能跑通 host/bootstrap 级 smoke。
- 真实设备失败时，日志能明确失败落点属于宿主、平台、还是执行层。

**完成判定：**
- MLU bring-up 从“黑盒失败”变成“可分阶段定位”。

- [ ] **Unit 5: 以 MLU 第一次真实生成为验收口径，收敛执行层剩余缺口**

**目标：** 在宿主已稳定的前提下，用真实机器的第一次 `qwen3 dense` 生成作为执行层最终验证。

**对应需求：** R4, R5, R8, R8.1, R8.2

**文件：**
- 修改：`vllm_xpu/attention/backends/flash_attn.py`
- 修改：`vllm_xpu/ops/infinicore/attention.py`
- 修改：`vllm_xpu/oot.py`
- 必要时修改：`vllm_xpu/runtime/infinicore.py`
- 新建或修改测试：`tests/integration/test_mlu_qwen3_dense_smoke.py`

**方案：**
- 只根据真实 bring-up 错误补最小执行缺口：
  - KV cache 布局/slot mapping
  - attention metadata builder 和实际 `vllm` 调用点的 shape 契合
  - OOT layer 对单卡 unquantized qwen3 dense 的覆盖完整性
- 不在这一单元顺便扩充多模型、多输入协议或采样特性。

**测试场景：**
- `examples/basic.py --model ...` 在目标 MLU 机器上能完成至少一次非 streaming 单 prompt 生成。
- 若失败，失败必须能归类到明确的未覆盖接缝，而不是宿主导入或平台选择问题。

**完成判定：**
- `qwen3 dense` 首次真实生成成功，且路径确认为 `vllm_xpu + Infinicore`，而不是 `vllm` 内建 CPU/XPU fallback。

## 风险与应对

- **风险：monkey patch 只能覆盖主进程，worker 子进程仍在 import `vllm` 时失败。**
  - 应对：先显式 patch 主入口；若 bring-up 表明子进程同样需要，则升级到 `sitecustomize` 或统一 launcher 注入。

- **风险：设备定制 `torch` 缺失更多私有 API，不止 `GraphCaptureOutput`。**
  - 应对：将 host compat 设计成集中 shim 层，按“可选 patch”模式系统化处理，而不是按单点补丁修一个报一个。

- **风险：宿主兼容修复后，执行层仍暴露更多 shape/layout 问题。**
  - 应对：保持 `basic.py` + debug 脚本分层验证，先把失败归类，再收缩执行层补丁。

- **风险：monkey patch 覆盖面失控，变成维护另一套隐式宿主行为。**
  - 应对：只补 import-time 必需兼容点；一旦 patch 开始触及模型执行语义，就停止并升级为显式上游 patch 决策。

## 测试与验收策略

### Characterization / Host

- `patch_for_vllm_import()` 之后，`import vllm` 在 `empty` 模式宿主环境中成功
- `import vllm_xpu` 后，plugin 注册不报错
- 平台选择明确命中 `XPUPlatform`

### Integration / Local

- `tests/unit/test_vllm_registration.py`
- `tests/unit/platforms/test_xpu_platform.py`
- `tests/integration/test_single_card_bf16_smoke.py`
- `tests/integration/test_basic_example.py`
- 新增 `empty` 模式相关 smoke

### Bring-up / Target MLU

- 安装：`VLLM_TARGET_DEVICE=empty` editable host + `pip install -e vllm-triton`
- 验证：`python examples/debug_host_bootstrap.py`
- 验证：`python examples/debug_platform_selection.py`
- 验证：`python examples/basic.py --model <qwen3-dense>`

## 依赖与顺序

1. Unit 1 先明确宿主基线，否则后续计划对象会继续漂移。
2. Unit 2 先用 host compat 消除 `vllm` import-time torch 硬依赖，这是 bring-up 的第一阻塞。
3. Unit 3 在宿主可导入后，确保插件和平台选择稳定。
4. Unit 4 建立分层调试闭环，避免把执行层问题和宿主问题混在一起。
5. Unit 5 才用真实 MLU 机器驱动执行层最终收敛。

## 交付物

- 一条受支持的 `empty-target` 宿主安装与 bring-up 路线
- 一组位于 `vllm_xpu` 内、发生在 `import vllm` 之前的最小 host compat 补丁点
- 一组能定位 host/platform/execution 三层问题的调试入口
- 一次真实 MLU 上的 `qwen3 dense` 生成验收
