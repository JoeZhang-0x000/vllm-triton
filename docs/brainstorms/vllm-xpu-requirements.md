---
date: 2026-04-01
topic: vllm-xpu
---

# vLLM XPU

## Problem Frame
当前目标是参考 `vllm-mlu` 的 out-of-tree 接入方式，设计并实现一个更通用的 `vllm-xpu` 插件，使用户在仅安装 `vllm-cpu` 的前提下，通过 Python 侧 `import` 插件即可启用单卡 GPU / 类 GPU 设备上的 BF16 推理能力。

这个项目要解决的问题不是为某一种设备做深度定制，而是定义一个足够小、足够稳定的通用 `xpu` 平台骨架：
- 默认提供一套通用 Triton 算子库，先满足单卡 BF16 推理
- 允许后续按设备逐步替换部分算子组，而不是复制一整套专有后端
- 尽量避免 `vllm-mlu` 风格的大面积 hijack，保持第一性原理和结构简洁

## Requirements
- R1. 提供一个单一的 `xpu` 平台插件，对外暴露统一产品形态，而不是拆成多个一阶段子包。
- R2. 第一阶段仅支持单卡、BF16 推理。
- R3. 第一阶段用户入口仅限 Python：用户安装 `vllm-cpu` 后，通过 `import vllm_xpu` 即可启用 `xpu` 能力，并继续使用标准 `vllm.LLM(...)` 入口。
- R4. `xpu` 平台必须依赖一个最小 runtime adapter 合同，而不是假定所有设备天然兼容。该合同至少覆盖设备设置、同步、显存信息查询、设备标识等基础能力。
- R5. 框架必须内置一套默认 Triton 算子库，作为第一阶段的通用计算实现。
- R6. 框架必须支持按“算子组”进行替换，而不是只支持替换单个细粒度算子，也不是要求后续设备接管整条执行路径。
- R7. 算子组替换必须支持部分覆盖。未被设备定制库覆盖的算子组，继续回退到默认 Triton 实现。
- R8. 第一阶段要优先复用 vLLM 已有能力与插件机制，尽量缩小对 vLLM 核心路径的改写范围。
- R9. 第一阶段明确不支持量化、通信、多卡并行、MoE 及其相关专用算子路径。
- R10. 设计必须为后续设备定制化保留稳定扩展点，但不能因为未来可能性而引入当前不需要的复杂抽象。

## Success Criteria
- 用户在只安装 `vllm-cpu` 和 `vllm-xpu` 插件的情况下，可以通过 Python 入口完成至少一条单卡 BF16 推理链路。
- `xpu` 平台核心只依赖一个小型 runtime adapter 合同，而不是将 CUDA 或 MLU 语义硬编码进平台层。
- 默认 Triton 算子库可以独立支撑第一阶段运行；后续设备库可以只替换部分算子组并与默认实现共存。
- 第一阶段代码结构清晰可解释，平台层、runtime adapter、默认算子库、设备定制扩展点之间职责分离明确。
- 接入方案相比 `vllm-mlu` 明显减少大面积 hijack 和设备专有分叉。

## Scope Boundaries
- 不做量化，包括 AWQ、GPTQ、FP8、SmoothQuant 等派生路径。
- 不做多卡、通信、并行调度、Ray 适配、分布式执行器扩展。
- 不做 MoE、EP、共享专家、相关通信算子与调度逻辑。
- 不要求第一阶段兼容 `vllm serve` 或原生 CLI。
- 不要求“任意设备零适配直接运行”；设备侧仍需满足最小 runtime adapter 合同。
- 不在 brainstorm 阶段定义具体代码布局、类名、注册 API 细节或算子实现细节。

## Key Decisions
- 单一 `xpu` 平台优先于多子后端拆分：先把产品形态做简单，用户只需要理解一个插件。
- Python-only 入口优先于 CLI 兼容：先压低集成复杂度，避免为了接管启动链路而扩大 hijack 面。
- 最小 runtime adapter 优先于“天然兼容”假设：通用性来自稳定合同，不来自忽略设备差异。
- 按算子组替换优先于按单算子或整路径替换：这是当前最平衡的扩展粒度，既保留演进空间，又避免注册表碎片化或重新走向专有后端。
- 部分覆盖并回退到默认 Triton 优先于全量接管：这能降低后续设备接入成本，也更符合框架化目标。
- 第一阶段默认 Triton 算子库优先于一开始就做多套专有实现：先把通用路径做通，再逐步替换热点能力。
- 简洁优先于“未来功能预埋”：只有与单卡 BF16 推理直接相关的抽象才应进入第一阶段。

## Dependencies / Assumptions
- vLLM 的 out-of-tree plugin 机制足以支撑 `xpu` 平台注册与最小范围的通用插件初始化。
- 第一阶段目标设备至少能通过某种方式承载 PyTorch tensor，并执行所需 Triton kernel。
- 默认 Triton 算子库覆盖的模型能力范围需要在 planning 阶段进一步收敛到一个最小可跑通集合。

## Outstanding Questions

### Resolve Before Planning
- 无

### Deferred to Planning
- [Affects R4][Technical] `xpu` runtime adapter 的最小接口面具体包含哪些方法、返回值和错误语义。
- [Affects R5][Technical] 第一阶段默认 Triton 算子库需要覆盖哪些最小算子组，才能支撑目标模型集合。
- [Affects R6][Technical] 算子组注册、选择、部分覆盖和回退的最简机制是什么。
- [Affects R8][Needs research] 复用 vLLM 插件机制时，哪些改动可以通过平台注册解决，哪些仍需要有限 hijack。
- [Affects R2][Needs research] 第一阶段最小支持的模型集合应如何收敛，以避免为了广泛兼容而破坏简洁性。

## Next Steps
→ `/prompts:ce-plan` for structured implementation planning
