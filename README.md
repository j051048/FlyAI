# FlyAI / Shard

[中文](#当前版本中文) · [English](#current-version-english) · [文档导航 / Documentation](docs/DOCUMENTATION_INDEX.md)

## 当前版本（中文）

文档核对日期：**2026-10-09**；本轮基于 [a99cb3c](https://github.com/j051048/FlyAI/commit/a99cb3c2274042d48373cf2abb09c518447843cf)，增加默认关闭的投机控制、全环状态缓存和有界专家预填充流水。
FlyAI 基于 Shard 的分布式推理框架演进。上游项目为 [leyten/shard](https://github.com/leyten/shard)，
其 c0mpute 集成背景和历史实验保留在仓库记录中。

项目目标是让任意贡献者登记 GPU 能力，由网络优先寻找实测低延迟的近邻或同区域节点，
按模型、显存、主机内存和运行配置组成推理环。相邻节点允许同机或同公网 IP；
GPU 身份、共享 RAM／锁页预算和实际链路决定放置。节点注册开放，执行需要兼容模型、
有效校准、资源租约和真实暖机。

### 已接入的能力与边界

| 领域 | 当前实现 | 验证与边界 |
|---|---|---|
| 开放注册 | Ed25519 签名 offers、TTL／序列防重放、模型 cohort | 签名说明报告者身份；资源数据仍需实际校准 |
| 组环与异构切层 | 区域优先、有向实际路线、联合 head／tail、精确可执行模板 | 搜索有界，性能预测不能替代集群实测 |
| 资源所有权 | 持久 GPU／共享 RAM／pinned 预算租约、fencing、进程驻留生命周期 | 未确认进程退出时继续保留资源占用 |
| GPT-OSS 服务 | 原 `specpipe` 运行时接入签名会话、认证 HTTP/SSE、租户队列和多环 | 当前生产适配器为 greedy text chat；单环一个串行 worker |
| V4 服务 | DSpark、可选本地专家 RAM／GPU 缓存、受控预取、KV 工作区、查询分块预填充 | 每项优化有数值／资源门控；新增路径 GPU 速度线待验收 |
| V4 权重准备 | 有界 HF 流式转换、按层/角色分发、局部分片完整哈希、磁盘/RAM 准备租约和原子发布 | 配置准备源后自动执行；只支持已声明的原 V4-Flash native ABI，真实集群仍须验收 |
| V4 状态优化 | 请求边界选择校准投机配方、签名全环恢复、保序专家预填充 DMA 流水 | 默认关闭；完全相同前缀可复用，扩展前缀每请求完整参考预填充验证；GPU 收益待实测 |
| 多环与恢复 | 多环并行、版本绑定、取消／期限、原请求重放和前缀核对 | 持续批处理、持久协调器 HA 和跨节点专家副本未作为生产能力提供 |
| 收据 | 实际激活字节承诺、签名、nonce、层覆盖和相邻链核对 | 收据不是完整模型诚实计算的密码学证明；计算节点可见其处理的激活 |

最近一轮相关 CPU／socket／HTTP 回归：**1839 通过、3 项环境条件跳过**；44 项 GPU 标记用例未纳入本轮。
缓存遍历器收尾修正后，另跑相关专项回归：**137 通过**，4 项 GPU 标记用例未执行；两组结果不相加。
新增两项 CUDA 缓存／图指针用例在本机因无 CUDA 条件跳过，不算 GPU 验收。
这是所选回归集的结果，真实 MXFP4 GPU 执行和新增 V4 四卡／六卡持续性能需要集群重测。

### 使用当前入口

* GPT-OSS 下载、模型身份、清单部署和测量：[GPT_OSS_PRODUCTION](docs/GPT_OSS_PRODUCTION.md)。
* 开放注册、租约、精确校准和多环生命周期：[OPEN_INFERENCE_NETWORK](docs/OPEN_INFERENCE_NETWORK.md)。
* V4 严格生产组环：[V4_CLUSTER_DEPLOY_GUIDE](docs/V4_CLUSTER_DEPLOY_GUIDE.md)。
* V4 自检、strict SSH 部署与性能复测：[V4_OPERATIONS](docs/V4_OPERATIONS.md)。
* 三项可选状态优化、配置与验证边界：[V4_STATEFUL_OPTIMIZATIONS](docs/V4_STATEFUL_OPTIMIZATIONS.md)。
* 小磁盘节点、流式分片与请求边界换环：[WEIGHT_PREPARATION](docs/WEIGHT_PREPARATION.md)。
* V4 固定速度线与原始证据：[V4_BENCHMARK](docs/V4_BENCHMARK.md)。
* 全部当前规范、研究与历史记录：[DOCUMENTATION_INDEX](docs/DOCUMENTATION_INDEX.md)。

完整仓库中的生产服务入口是：

```sh
# 配置、节点模板、路由和模型目录需按对应指南准备。
SHARD_TRANSPORT=libp2p SHARD_RECEIPTS=1 \
  python engines/gpt_oss/network_service.py --config network.json --auth-file tenants.json
python engines/deepseek_v4/v4_network_service.py --config network.json --auth-file tenants.json
```

严格模式使用 `shard-pipeline-plan/1`、节点与协调器既有签名身份和完整模型 cohort。
旧手工实验通过明确的 `--legacy-protocol` 使用兼容路径；V4 的旧单环／静态环池
`v4_gateway.py` 入口仍需按其指南的兼容合同配置。公网控制及 HTTP 服务要求 TLS。
现有 libp2p sidecar 负责连接、加密、NAT 穿透与中继；服务适配器使用已配置的实际路线。

### 仓库结构

```text
shard/       通用协议、offers、租约、规划、会话、资源、收据、队列与 HTTP 服务
engines/     deepseek_v4/、minimax_m25/、kimi_k3/ 的模型实现；gpt_oss/ 服务适配器
phase0/      GPT-OSS specpipe／FastVerify、旧兼容入口、部署与验收工具
vendor/      保留来源和许可证的参考数学实现
sidecar/     Go libp2p 传输与身份服务
research/    研究探针和原型
docs/        当前指南、接口规范及有日期的历史证据
```

GPT-OSS 的模型运行时仍在 `phase0/`，新增服务适配器位于 `engines/gpt_oss/`。
共享能力置于 `shard/`；模型引擎不相互引用。`ModelRuntime` 的全模型统一仍在演进，
已有多个实际后端不意味着任意新架构可直接运行。训练和通用计算属于长期方向。

### 历史实测与论文

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21178430.svg)](https://doi.org/10.5281/zenodo.21178430)

上游技术报告：[Sharded Inference of a 229B-Parameter MoE over the Public Internet at Interactive Speed](docs/paper/main.pdf)。
下表是对应日期与配置下的历史记录，原始摘要、收据及研究条件见链接；不代表当前版本所有工作负载的保证。

| 历史记录 | 报告值 | 对应证据 |
|---|---|---|
| GPT-OSS-120B，2026-06 消费卡 WAN 实验 | 约 40 tok/s；部分复制／长上下文实验有更高接受收益 | [2026-06-19 收据](docs/receipts/gpt-oss-120b-wan-20260619.json)、[WAN 研究](docs/research/wan-speculative-decoding.md) |
| GLM-5.2，2026-06 七 GPU 研究拓扑 | 约 30 tok/s | [原始汇总](docs/receipts/glm52-nvfp4-wan-20260618.json)、[研究记录](docs/research/glm-5.2-on-consumer-blackwell.md) |
| V4-Flash，2026-08 六张 5090 原路径 | 汇总报告 30.15 warm tok/s | [实验与限制](docs/V4_FLASH_ENGINE.md)、[负载矩阵](docs/receipts/v4-flash-matrix-20260802.json) |

签名收据与记录中的计时／硬件声明各有验证范围，不能把所有汇总文件当作独立硬件证明。
GPT-OSS 旧 `recv/round` 混合等待时间，旧 full-accept／resume 吞吐口径已修正；历史数字保留原口径。
V4 新增路径继续以 **4×5090 ≥40、6×5090 ≥30 valid output tok/s**、固定协议和同请求对照为验收目标。

## Current version (English)

Documentation reviewed **2026-10-09**, based on [a99cb3c](https://github.com/j051048/FlyAI/commit/a99cb3c2274042d48373cf2abb09c518447843cf) plus opt-in request policy, full-ring state snapshots and bounded expert-prefill pipelining.
FlyAI evolves the Shard inference framework. The [upstream repository](https://github.com/leyten/shard),
c0mpute integration history and dated experiments remain attributed in the documentation.

Contributors can register signed GPU offers without an operator allowlist. Execution requires a compatible
model cohort, measured resources, an executable calibration template, a live lease and verified warmup.
Placement prefers nearby or complete regional rings and expands when necessary. Co-located distinct GPUs
are allowed; shared host budgets and actual directed routes determine suitability.

The current implementation includes signed pipeline sessions, remote coordinator exclusion, persistent
resource ownership, exact calibrated heterogeneous placement, authenticated HTTP/SSE, globally bounded
multi-ring queues, request cancellation and original-prompt replay with prefix checking. Each ring has a
serial worker; separate rings execute concurrently. Continuous batching and durable coordinator HA remain
separate work. Receipts bind signed activation commitments, job freshness and layer coverage; they do not
constitute a cryptographic proof of honest model execution or conceal activations from computing nodes.

GPT-OSS retains its existing `phase0/specpipe.py` runtime with a new `engines/gpt_oss/` serving adapter.
V4 retains its dedicated engine, DSpark and gated optional local RAM/expert-cache/KV optimizations.
The [stateful optimizations](docs/V4_STATEFUL_OPTIMIZATIONS.md) select approved recipes at request boundaries,
restore exact-repeat prefixes through signed all-stage barriers, and pipeline bounded local expert DMA.
Extended-prefix reuse requires a full-prefill reference shadow for each request; this path does not claim
a general multi-turn speedup. All three features remain disabled by default pending GPU acceptance.
V4 native artifacts now have packing-independent model identity, bounded HF conversion, verified
per-stage acquisition, shared disk/RAM preparation reservations and atomic publication. Configured
same-cohort alternatives can replace failed rings at request boundaries; existing jobs retain their bindings.
MiniMax-M2.5 and Kimi-K3 have their existing model-specific implementations. New model architectures
require their own implementation and measurements; the `ModelRuntime` consolidation remains in progress.

Start with [GPT-OSS deployment](docs/GPT_OSS_PRODUCTION.md), [open-network contracts](docs/OPEN_INFERENCE_NETWORK.md),
[V4 deployment](docs/V4_CLUSTER_DEPLOY_GUIDE.md), [weight preparation](docs/WEIGHT_PREPARATION.md)
and the [documentation index](docs/DOCUMENTATION_INDEX.md).
Strict production services are `engines/gpt_oss/network_service.py` and
`engines/deepseek_v4/v4_network_service.py`; they require full checkouts, pinned signing identities,
model content identities, calibrated local stage templates and configured inference routes. Explicit
legacy CLI modes retain older experiments. Public control/HTTP listeners require TLS, while existing
sidecars own peer transport and NAT handling.

The latest selected CPU/socket/HTTP regression set passed **1839 tests with 3 environment-dependent skips**; 44 GPU-marked cases were deselected.
After the final bounded-iterator correction, **137 targeted tests passed**, with 4 GPU-marked cases deselected; these overlapping suites are not additive.
Two additional CUDA cache/graph-pointer gates were conditionally skipped on this CPU-only runtime.
This is local implementation evidence. Historical GPU numbers above retain their original workload,
recipe and timing scope. Actual native MXFP4 GPU execution and new V4 four-card >=40 / six-card >=30 tok/s
acceptance still require live cluster verification. Fixed-K adaptive depth is available; mixed-K experiments
charge their full greedy-control overhead and cannot use a single-shape prediction as a certified speed result.

## License

[Apache License 2.0](LICENSE) © 2026 leyten. Vendored references retain their own licenses and provenance.
