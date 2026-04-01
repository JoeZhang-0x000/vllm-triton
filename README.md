# vllm-xpu Foundation

这个仓库用于实现一个尽量简洁的 out-of-tree `vllm_xpu` 插件基础框架。

当前阶段已完成的内容：
- Python 包骨架与 vLLM plugin entry points
- package bootstrap
- runtime adapter registry
- operator provider registry

当前阶段尚未完成的内容：
- 真实 XPU platform / worker 集成
- 默认 Triton provider
- 单卡 BF16 端到端推理路径
