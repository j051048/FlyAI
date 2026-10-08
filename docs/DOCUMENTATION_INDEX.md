# 当前文档导航与历史证据

核对日期：2026-10-08；实现基线：`a6e96e3` 加本地权重准备/换环及 P0–P2 运行改造。本页区分操作合同、接口规范、
有日期的实验记录和参考实现来源。历史性能数字按原实验条件保留；新版计时、
会话或资源合同需要新的原始运行证据。

## 从哪里开始

| 目的 | 当前入口 |
|---|---|
| 了解项目与已实现边界 | [README](../README.md)、[中文说明](../README_CN.md)、[实现状态](../STATE.md) |
| 部署 GPT-OSS、下载校验、严格会话与重测 | [GPT_OSS_PRODUCTION](GPT_OSS_PRODUCTION.md) |
| 部署 V4 严格服务 | [V4_CLUSTER_DEPLOY_GUIDE](V4_CLUSTER_DEPLOY_GUIDE.md)、[V4_GATEWAY](V4_GATEWAY.md) |
| V4 运行自检、strict SSH 部署、健康诊断和单变量复测 | [V4_OPERATIONS](V4_OPERATIONS.md) |
| 小磁盘节点、流式转换、权重准备与请求边界换环 | [WEIGHT_PREPARATION](WEIGHT_PREPARATION.md) |
| 任意节点注册、区域组环、租约与多环 | [OPEN_INFERENCE_NETWORK](OPEN_INFERENCE_NETWORK.md) |
| 资源准入与部署检查 | [RESOURCE_CONTRACT](RESOURCE_CONTRACT.md)、[DEPLOY_READINESS](DEPLOY_READINESS.md) |
| 区分预测、真实测量与速度验收 | [V4_BENCHMARK](V4_BENCHMARK.md)、[RUNTIME_METRICS](RUNTIME_METRICS.md) |
| 复核 2026-10-08 六卡实测与历史口径 | [V4_FIELD_REVIEW_20261008](V4_FIELD_REVIEW_20261008.md) |
| 理解收据、身份与隐私边界 | [PROOF](PROOF.md)、[V4_TRUST_BOUNDARIES](V4_TRUST_BOUNDARIES.md) |

当前严格生产适配器为 `engines/gpt_oss/network_service.py` 和
`engines/deepseek_v4/v4_network_service.py`，使用完整仓库、模型 cohort、
校准模板、签名部署清单、资源租约及配置好的真实路线。GPT-OSS 原模型运行时
仍在 `phase0/specpipe.py`。V4 旧 `v4_gateway.py` 单环／静态环池入口和旧实验
CLI 有各自兼容合同，不能仅更换命令名称就认为已经启用严格会话。

本轮选定 CPU/socket/HTTP 回归为 1620 通过、3 项环境条件跳过，41 项 GPU 标记未选入。
新增代码的 GPU 吞吐与长时间稳定性仍需实测。V4 四卡 >=40、六卡 >=30 valid
output tok/s 仍是固定验收目标；`v4_acceptance.py` 的模拟结果不通过该硬件门槛。

## 当前规范与操作指南

| 文档 | 维护范围 |
|---|---|
| [ARCHITECTURE](ARCHITECTURE.md) | 协议骨架、模型后端、控制面及信任边界 |
| [MODEL_RUNTIME](MODEL_RUNTIME.md) | ModelRuntime 接口、实际后端和继续统一的部分 |
| [RESOURCE_CONTRACT](RESOURCE_CONTRACT.md) | measured byte budgets、共享 RAM/pinned、硬件未知值 |
| [WEIGHT_PREPARATION](WEIGHT_PREPARATION.md) | 局部分片身份、共享磁盘/准备 RAM 预算、自动准备与请求边界切换 |
| [OPEN_INFERENCE_NETWORK](OPEN_INFERENCE_NETWORK.md) | offers、cohort、执行模板、租约、区域与生命周期 |
| [NETWORK](NETWORK.md) | 现有 sidecar、传输连接和网络假设 |
| [COLOCATION_POLICY](COLOCATION_POLICY.md) | 同机、同 IP 参与与实际资源/路由检查 |
| [HETERO_DEVICES](HETERO_DEVICES.md) | 异构放置与设备能力边界 |
| [ADMISSION_SPEC](ADMISSION_SPEC.md) | 计算参与及外部消费网络的准入/结算接口范围 |
| [INTEGRATION](INTEGRATION.md) | engine 与消费控制面集成方向 |
| [DEPLOY_READINESS](DEPLOY_READINESS.md) | 实际就绪条件与待验收项 |
| [LAUNCH](LAUNCH.md) | 现有启动工具及兼容迁移 |
| [PROOF](PROOF.md) | 签名、覆盖、链、挑战和未提供的诚实计算证明 |
| [RUNTIME_METRICS](RUNTIME_METRICS.md) | 收据观察、分段计时与独立 RTT 的范围 |
| [ROADMAP](ROADMAP.md) | 当前代码、硬件验证与未来工作分别维护 |
| [GPT_OSS_PRODUCTION](GPT_OSS_PRODUCTION.md) | GPT-OSS 严格部署、下载、测量和投机调优 |
| [V4_CLUSTER_DEPLOY_GUIDE](V4_CLUSTER_DEPLOY_GUIDE.md) | V4 当前操作步骤 |
| [V4_OPERATIONS](V4_OPERATIONS.md) | 共享运行初始化、caller-local 隧道、签名运行观察及独立链路探测 |
| [V4_GATEWAY](V4_GATEWAY.md) | HTTP/SSE、租户/模型路由、不同入口合同 |
| [V4_BENCHMARK](V4_BENCHMARK.md) | 固定协议、原始证据和 GPU 速度线 |
| [V4_NEXT_PHASE](V4_NEXT_PHASE.md) | 当前 V4 集成与下一阶段验证 |
| [V4_PRODUCTION_OPTIMIZATIONS](V4_PRODUCTION_OPTIMIZATIONS.md) | 运行时优化开关及门控 |
| [V4_HYBRID_RUNTIME](V4_HYBRID_RUNTIME.md) | GPU/RAM 双资源设计和各里程碑实际状态 |
| [V4_TRUST_BOUNDARIES](V4_TRUST_BOUNDARIES.md) | 注册、执行身份、收据和激活可见性 |
| [MLX_RUNTIME](MLX_RUNTIME.md) | Apple Silicon 后端的独立范围 |
| [MiniMax 部署](../phase0/DEPLOY_M25.md) | 既有 M2.5 路径，不能套用 GPT-OSS/V4 新握手 |
| [Sidecar](../sidecar/README.md) | Go libp2p 传输工具与运行参数 |

## 模型实验与阶段性验证

以下文档包含原实验设计、特定配置结果和阶段判断。其日期、模式、模型和
数值门控需一起阅读；当前状态与新命令通过文档内的更新说明和上方指南导航。

| 文档 | 记录主题 |
|---|---|
| [M25_ENGINE](M25_ENGINE.md) | M2.5 引擎演进及有日期的测量日志 |
| [V4_FLASH_ENGINE](V4_FLASH_ENGINE.md) | 原六卡 V4 跑环和模型机制 |
| [V4_FOUNDATION_VALIDATION](V4_FOUNDATION_VALIDATION.md) | 基础数学/权重/切层验证范围 |
| [V4_FULL_STACK](V4_FULL_STACK.md) | 特定全链路实验与验收边界 |
| [V4_PIPELINED_SPEC](V4_PIPELINED_SPEC.md) | 流水线投机实现及实验 |
| [V4_PIPELINE_EFFICIENCY](V4_PIPELINE_EFFICIENCY.md) | 流水线效率研究 |
| [V4_PERF_ROUND_2](V4_PERF_ROUND_2.md) | 第二轮性能实验 |
| [V4_NGRAM_PROPOSER](V4_NGRAM_PROPOSER.md) | proposer 实验和接受收益 |
| [V4_FILL_ECONOMICS](V4_FILL_ECONOMICS.md) | 在途窗口、浪费和收益 |
| [V4_MULTIBLOCK_VERDICT](V4_MULTIBLOCK_VERDICT.md) | 多块投机实验结论 |
| [V4_TREE_VERDICT](V4_TREE_VERDICT.md) | tree 实验结论 |

## 历史研究与原始报告

| 文档 | 日期/用途 |
|---|---|
| [WAN speculative decoding](research/wan-speculative-decoding.md) | 2026-06 研究日志；旧延迟/吞吐口径 |
| [GLM-5.2 consumer Blackwell](research/glm-5.2-on-consumer-blackwell.md) | 特定七 GPU 拓扑实验 |
| [Weight fetch validation](research/step3-weight-fetch-validation.md) | 早期下载/manifest 校验及其当时发现 |
| [M25 lever stack](research/m25-lever-stack-verified-20260716.md) | 2026-07-16 预测和 07-18 后续门控，不能把预测当实测 |
| [Reasoning baseline](receipts/m25-honest-reasoning-baseline-20260629.md) | 2026-06-29 特定 M2.5 基线 |
| [EAGLE on-engine](receipts/m25-eagle-onengine-20260629.md) | 2026-06-29 失败与诊断记录 |
| [Usability report](receipts/m25-usability-report-20260702.md) | 2026-07-02 特定网络与负载 |
| [Good-ring receipt](receipts/m25-goodring-receipt-20260703.md) | 2026-07-03 运行记录 |
| [Real-ring loop](receipts/m25-realring-loop-20260707.md) | 2026-07-07 集成与外部结算演示范围 |
| [Warm-ring validation](receipts/m25-warmring-validation-20260707.md) | 2026-07-07 测量和运行条件 |
| [Historical fleet](../FLEET_STATE.md) | 六月租赁记录；当前实例状态需运营者查询 |
| [Implementation journal](../STATE.md) | 当前状态摘要及保留的六月日志 |

JSON 收据与报告保留原始字节，当前校准/bench/soak 的验证规则不能反向认证
旧格式的所有硬件或速度声明。签名证明签署与内容绑定，实际计算和计时各有范围。

## 参考实现来源

[DeepSeek reference provenance](../vendor/deepseek_v4_ref/PROVENANCE.md) 保留上游快照、
许可证与数学来源。参考数学文件保持原样；现有适配器路径以 `engines/` 和
当前部署指南为准。

## 维护约定

当前命令以本地 CLI/代码为准。改入口、协议、校准字段或就绪条件时，同时更新
对应当前指南和本页。历史原始数值、收据与失败记录保留日期和范围；新增复测单独
记录。运行 `python tools/check_references.py` 检查全仓 Markdown 本地文件链接和
workflow 路径；该检查跳过代码示例与外链，不认证外部页面内容或 Markdown anchor。
