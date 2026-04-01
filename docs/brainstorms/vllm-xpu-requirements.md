---
date: 2026-04-01
topic: vllm-xpu
---

# vLLM XPU

## Problem Frame
当前目标是重新收敛 `vllm-xpu` 的第一阶段方向：不再把“默认 Triton fallback”作为核心路径，而是把 `Infinicore` 作为唯一默认执行基座，用来承接那些不具备 PyTorch 原生能力的 XPU 设备。

这个项目要解决的问题也因此发生了变化。第一阶段不再优先追求“任意设备后端都能平等接入”的通用框架，而是先定义一条足够小、足够稳定、真实可跑的产品路径：
- 用户仍然通过统一的 `xpu` 插件入口使用 vLLM
- `vllm_xpu` 内部默认统一到 `Infinicore` 的 tensor / device / operator 语义
- 目标是在非 PyTorch-native XPU 上跑通 `qwen3 dense` 的单卡 BF16 最小闭环
- `torch` 只允许保留在必须与 vLLM 对接的兼容边界上，而不是继续作为 `vllm_xpu` 内部执行真相

这次重规划的关键不是换一个 fallback 名字，而是明确第一阶段真正要验证的东西：`vllm_xpu` 是否能把 vLLM 宿主和 `Infinicore` 执行世界可靠地接起来。

## Requirements
- R1. 提供一个单一的 `xpu` 平台插件，对外暴露统一产品形态，而不是拆成多个一阶段子包。
- R2. 第一阶段仅支持单卡、BF16 推理。
- R3. 第一阶段用户入口仅限 Python：用户安装 `vllm-cpu` 后，通过 `import vllm_xpu` 即可启用 `xpu` 能力，并继续使用标准 `vllm.LLM(...)` 入口。
- R4. 第一阶段默认执行基座固定为 `Infinicore`；`vllm_xpu` 内部不再以 Triton fallback 或 PyTorch device/runtime 语义作为主路径。
- R5. 第一阶段必须以 `Infinicore` 提供 `qwen3 dense` 最小推理闭环所需的基础算子与执行能力。
- R6. 第一阶段要明确区分“vLLM 宿主兼容边界”和“`vllm_xpu` 内部执行边界”；除必要适配外，`torch` 不能继续渗透进 `vllm_xpu` 内部执行路径。
- R7. 第一阶段要优先复用 vLLM 已有 plugin / platform 能力，尽量缩小对 vLLM 核心路径的改写范围。
- R8. 第一阶段成功标准不是“抽象上支持多后端”，而是“在非 PyTorch-native XPU 上真实跑通一个 `qwen3 dense` 单卡 BF16 闭环”。
- R8.1. 第一阶段必须提供一个仿照 vLLM `basic.py` 风格的最小本地调用示例脚本，支持命令行参数 `--model`，并内置一条最简单的固定 prompt 作为验收入口。
- R8.2. 第一阶段的模型目标收敛为“`qwen3 dense` 架构路径”，不绑定某个固定 checkpoint 或参数规模。
- R8.3. 第一阶段以“先真实跑通”为优先，不强求主执行过程的边界纯度；在核心执行链路落到 `Infinicore` 的前提下，可以暂时保留必要的 `torch` 兼容对象与中间桥接。
- R8.4. 第一阶段生成能力目标应尽量贴近 vLLM 常规文本生成体验，而不是只做单 prompt greedy demo；但该目标默认仅限文本生成主路径，不自动扩展到多模态、工具调用或其他高级能力。
- R8.5. 第一阶段文本生成入口只支持 plain text prompt，不要求支持 `chat template`、`messages` 或对话式输入协议。
- R8.6. 第一阶段不要求支持 streaming；首版验收与示例脚本仅覆盖非 streaming 文本生成。
- R8.7. 第一阶段非 streaming 文本生成只要求单 prompt 主路径，不要求 batching 或多 prompt 调度能力。
- R9. 第一阶段明确不支持量化、通信、多卡并行、MoE 及其相关专用算子路径。
- R10. 第一阶段不要求同时兼容多套默认执行基座；如后续要支持并列后端，应在 `Infinicore` 路径跑通之后再单独规划。

## Success Criteria
- 用户在仅通过 Python 入口启用 `vllm_xpu` 的前提下，可以在一个非 PyTorch-native XPU 环境中跑通 `qwen3 dense` 的单卡 BF16 最小生成链路。
- 仓库内提供一个最小本地示例脚本，用户可通过 `--model` 指定本地或模型标识，并用固定 prompt 复现一次真实生成。
- 首版需求面向 `qwen3 dense` 架构路径，而不是某个固定模型尺寸；planning 阶段应据此提炼最小公共算子面。
- 首版允许采用“边跑通边收敛边界”的策略，而不是在实现前先把 `torch` / `Infinicore` 分界打磨到完全纯净。
- 首版生成目标不再局限于单 prompt greedy 演示，而是尽量贴近 vLLM 常规文本生成主路径。
- 首版用户输入面收敛为 plain text prompt，不包含 chat/messages 兼容目标。
- 首版生成模式收敛为非 streaming。
- 首版调度面收敛为单 prompt 主路径。
- `vllm_xpu` 内部的默认执行路径统一建立在 `Infinicore` 之上，而不是继续混合 Triton fallback 与 PyTorch runtime 语义。
- 与 vLLM 的对接边界清晰可解释：`torch` 只出现在必要兼容层，不作为 `vllm_xpu` 内部执行实现的默认依赖。
- 第一阶段代码结构清晰可解释，平台接入、宿主兼容边界、`Infinicore` 执行路径三者职责分离明确。
- 接入方案相比 `vllm-mlu` 和“默认 Triton fallback”方向，明显减少为了未来多后端而引入的过度抽象。

## Scope Boundaries
- 不做量化，包括 AWQ、GPTQ、FP8、SmoothQuant 等派生路径。
- 不做多卡、通信、并行调度、Ray 适配、分布式执行器扩展。
- 不做 MoE、EP、共享专家、相关通信算子与调度逻辑。
- 不要求第一阶段兼容 `vllm serve` 或原生 CLI。
- 不要求第一阶段兼容任意设备后端；第一阶段默认只服务 `Infinicore` 这一条执行路径。
- 不要求第一阶段保留并列的 Triton fallback。
- 不要求在第一阶段解决“完全摆脱 vLLM 对 torch 的宿主依赖”这一更大问题。
- 不在 brainstorm 阶段定义具体代码布局、类名、注册 API 细节或算子实现细节。

## Key Decisions
- 单一 `xpu` 平台优先于多子后端拆分：先把产品形态做简单，用户只需要理解一个插件。
- Python-only 入口优先于 CLI 兼容：先压低集成复杂度，避免为了接管启动链路而扩大 hijack 面。
- 第一阶段默认执行基座固定为 `Infinicore`：先把真实目标设备跑通，而不是继续为“未来可能的多后端”支付抽象成本。
- `Infinicore` 内部语义优先于 PyTorch 语义：`vllm_xpu` 内部应围绕 `Infinicore` 的 tensor / device / op 能力建模。
- `torch` 只作为 vLLM 宿主兼容边界存在：如果某处只是为了配合 vLLM 接口才需要 `torch` 语义，应把它压缩在边界层而不是向内部扩散。
- `qwen3 dense` 优先于广泛模型兼容：第一阶段只验证价值最高、最小可跑的模型闭环，不为了泛化而扩大范围。
- 首版跑通优先于边界洁癖：只要没有把 `torch` 再次扩张成内部默认执行真相，允许保留必要的过渡桥接。
- 首版体验优先对齐常规文本生成，而不是只做一个过窄的 greedy-only demo。
- 首版输入协议保持最小：先围绕 plain text prompt 跑通，再决定是否扩到 chat/messages。
- 首版不追求 streaming 体验，先把非 streaming 主路径跑通。
- 首版不追求 batching，对单 prompt 路径负责即可。
- 简洁优先于“未来功能预埋”：只有与 `Infinicore + qwen3 dense + 单卡 BF16` 直接相关的抽象才应进入第一阶段。

## Dependencies / Assumptions
- vLLM 的 out-of-tree plugin 机制足以支撑 `xpu` 平台注册与最小范围的通用插件初始化。
- `Infinicore` 已经能够为目标 XPU 提供一套足以支撑最小推理闭环的 tensor、device 和 operator 能力。
- 第一阶段目标设备不要求具备 PyTorch 原生能力，但必须能被 `Infinicore` 统一承接。
- `qwen3 dense` 的最小执行闭环仍需要在 planning 阶段进一步收敛为一个明确的算子与集成范围。

## Outstanding Questions

### Resolve Before Planning
- [Affects R4][Technical] `vllm_xpu` 内部哪些位置必须保留为 vLLM 的 `torch` 兼容边界，哪些位置应彻底切换到 `Infinicore` 语义。
- [Affects R5][Technical] 为跑通 `qwen3 dense` 最小闭环，`Infinicore` 必须覆盖的最小算子集合与执行能力边界是什么。
- [Affects R8.4][Product] “贴近 vLLM 常规文本生成体验”在第一阶段至少包含哪些用户可见生成能力，哪些应明确排除。

### Deferred to Planning
- [Affects R4][Technical] 当前 runtime / worker / model-runner 桥接层中，哪些抽象应保留，哪些会因为“唯一默认基座是 `Infinicore`”而被收缩或移除。
- [Affects R5][Technical] `Infinicore` provider 与 vLLM attention / layer 执行接缝的最小落点应该放在哪些 seam 上。
- [Affects R7][Needs research] 复用 vLLM plugin 机制时，哪些改动可以通过平台注册解决，哪些仍需要有限 hijack。
- [Affects R8.2][Needs research] `qwen3 dense` 架构路径的最小公共支持面应收敛到哪些具体层和路径，才能避免为了广泛兼容而破坏简洁性。
- [Affects R8.1][Technical] 示例脚本具体放在哪个路径，以及是否完全复用现有 `examples/offline_inference.py` 还是新建一个更贴近 vLLM `basic.py` 的入口。

## Next Steps
→ Resume `/prompts:ce-brainstorm` to resolve blocking questions before planning
