# FlyAI / Shard

[中文](#当前版本中文) · [English](#current-version-english) · [文档导航 / Documentation](docs/DOCUMENTATION_INDEX.md)

## 当前版本（中文）

文档核对日期：**2026-10-29**。本轮补全全环成本反馈、实际字节设备摘要、生产形状选环、持久异长批量状态、RAM 专家分段图和资格门控的增量前缀／精确 KV gather。新路径默认关闭，原模型 GPU、性能与稳定性仍待按[六项验收步骤](docs/V4_SIX_ACCEPTANCE.md)验证。前轮角色成本、本机激活、专家冷补集、MTP 图和训练工具见[已有六项合同](docs/V4_DUALGPU_IMPROVEMENTS.md)；原 B4、whole 回滚与独立 partial 批图门槛保留。
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
| 组环与异构切层 | 区域优先、有向实际路线、联合 head／tail、精确模板；可选尾延迟、成本和共同故障域目标 | 未知指标保留 unknown；组件 p95 压力预测不能替代集群实测 |
| 资源所有权 | 持久 GPU／共享 RAM／pinned 预算租约、fencing、进程驻留生命周期 | 未确认进程退出时继续保留资源占用 |
| GPT-OSS 服务 | 原 `specpipe` 运行时接入签名会话、认证 HTTP/SSE、租户队列和多环 | 当前生产适配器为 greedy text chat；单环一个串行 worker |
| V4 服务 | DSpark、可选本地专家 RAM／GPU 缓存、受控预取、KV 工作区、查询分块预填充 | 每项优化有数值／资源门控；新增路径 GPU 速度线待验收 |
| V4 权重准备 | 有界 HF 流式转换、按层/角色分发、局部分片完整哈希、磁盘/RAM 准备租约和原子发布 | 配置准备源后自动执行；只支持已声明的原 V4-Flash native ABI，真实集群仍须验收 |
| V4 状态优化 | 请求边界选择校准投机配方、签名全环恢复、保序专家预填充 DMA 流水 | 默认关闭；精确重复可复用，扩展前缀须完整原生资格和首轮/周期 canary；GPU 收益待实测 |
| V4 多卡执行 | 显式共享配方、实际图覆盖、数值/回滚诊断、原生内核探针、strict 宿主分组与有界接收缓冲 | 保留逐 GPU 身份和共享预算；四策略及故障矩阵已接入，GPU/速度资格待验，见 [多卡指南](docs/V4_MULTIGPU_EXECUTION.md) |
| V4 本机执行与草稿 | 角色成本、可信 CUDA IPC／共享内存路径、可选专家热冷补集、完整 committed-MTP 图、宿主预算与离线 draft 残差训练 | 默认关闭；收据仍哈希实际字节，target verifier 不变，真实 packed 模型与组合性能待验，见[本轮边界](docs/V4_DUALGPU_IMPROVEMENTS.md) |
| 5090 真机验收 | whole/eager 回滚状态、同机多卡、四模式短/长矩阵、持续 soak、原始证据归档 | [OpenClaw 操作步骤和验收入口](docs/V4_5090_ACCEPTANCE.md)；本地回归通过，真机资格待验 |
| 多环与恢复 | 多环并行、私有 SQLite 日志/owner fencing、原请求重放、SSE 游标、aging 与成本路由 | 原scalar lanes保留；独立`/2` greedy批合同、逐请求固定算形与可选partial图，非融合GEMM；GPU/FP4收益待验，无分布式HA |
| 回放质量 | 本机固定验证者的鲜活签名复算报告、按节点/cohort 限制新组环 | 不接受节点自报质量分，也不是硬件 attestation 或模型密码学证明 |
| 收据 | 实际激活字节承诺、签名、nonce、层覆盖和相邻链核对 | 收据不是完整模型诚实计算的密码学证明；计算节点可见其处理的激活 |

2026-10-12 六项多卡 P0–P2 的历史 **123 个相关测试文件**组合回归：**2081 通过、11 项环境条件跳过**，80 项 GPU 标记用例未选入。2026-10-21 冻结源码的137文件相关回归为 **2333通过、11跳过、92项GPU未运行**；详情见[改造记录](docs/V4_DEEPINFER_IMPROVEMENTS.md)。
覆盖真实原生 CPU 数学、strict socket、HTTP、SQLite、状态恢复及校准/经济证据；不代表 GPU 速度或全仓跨平台 CI。
完整 Windows 全仓尝试包含既有编码、POSIX 与模拟计时失败；代表用例已与 `070ccd6` 基线对照，不能将相关组合结果登记为全仓通过。
上一构建的组合回归为 1813 通过、8 项跳过、66 项 GPU 未选入。
前一轮执行改造的 111 文件组合为 1903 通过、11 跳过、69 GPU 未选入；这些专项不相加。
CI 报告复核后的部署／资源补充回归为 **218 通过、1 项 POSIX 用例在 Windows 跳过**；
同一进程生命周期套件在 WSL Ubuntu 22.04 上 **51 项通过**。各专项与组合结果不相加，
尚未把本机结果登记为 GitHub Python 3.12 全仓 CI 或真实 GPU 性能验收。
上一构建的 1839 项及后续 137 项重叠回归保留在历史状态记录中。
该构建另有两项 CUDA 缓存／图指针用例因本机条件跳过。这些是旧构建记录，不是本轮整体结果或 GPU 验收。
此前 P0–P2 的相关组合回归已完成；最新六项组合回归结果另行登记，历史数字不累加。真实 MXFP4 GPU 执行及 V4 四卡／六卡速度与持续运行仍需按新发布门槛实测。本轮按用户安排暂缓硬件验证。

### 使用当前入口

* GPT-OSS 下载、模型身份、清单部署和测量：[GPT_OSS_PRODUCTION](docs/GPT_OSS_PRODUCTION.md)。
* 开放注册、租约、精确校准和多环生命周期：[OPEN_INFERENCE_NETWORK](docs/OPEN_INFERENCE_NETWORK.md)。
* V4 严格生产组环：[V4_CLUSTER_DEPLOY_GUIDE](docs/V4_CLUSTER_DEPLOY_GUIDE.md)。
* V4 自检、strict SSH 部署与性能复测：[V4_OPERATIONS](docs/V4_OPERATIONS.md)。
* 三项可选状态优化、配置与验证边界：[V4_STATEFUL_OPTIMIZATIONS](docs/V4_STATEFUL_OPTIMIZATIONS.md)。
* 持久恢复、SLO/风险、请求调度、质量报告与发布门槛：[V4_SERVICE_HARDENING](docs/V4_SERVICE_HARDENING.md)。
* P0–P2 执行改造、投机校准热装、原生独立请求、增长前缀与健康环重组：[V4_PERFORMANCE_EXECUTION](docs/V4_PERFORMANCE_EXECUTION.md)。
* 小磁盘节点、流式分片与请求边界换环：[WEIGHT_PREPARATION](docs/WEIGHT_PREPARATION.md)。
* V4 固定速度线与原始证据：[V4_BENCHMARK](docs/V4_BENCHMARK.md)。
* B4数值修正、独立partial批图与scalar/eager/graph真机对照：[V4_BATCH_GRAPH_ACCEPTANCE](docs/V4_BATCH_GRAPH_ACCEPTANCE.md)。
* 本机 CUDA IPC、专家冷补集、完整 MTP 图与离线草稿训练：[六项合同](docs/V4_DUALGPU_IMPROVEMENTS.md)、[独立验收步骤](docs/V4_DUALGPU_ACCEPTANCE.md)。
* 全环成本、设备摘要、生产选环、持久异长批执行、专家分段图和增量 KV：[本轮六项原模型验收](docs/V4_SIX_ACCEPTANCE.md)。
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

Documentation reviewed **2026-10-29**. The latest default-off work connects all-stage round costs, actual-byte device hashing, production-shaped ring selection, persistent ragged request state, RAM-expert CUDA segments and qualified incremental-prefix/indexed-KV reuse. Native checkpoint, CUDA performance and long-running qualification remain pending; see [six-part acceptance](docs/V4_SIX_ACCEPTANCE.md). The preceding local-execution/draft work is described [here](docs/V4_DUALGPU_IMPROVEMENTS.md). Existing whole rollback, four decode modes and the separate batch contracts retain their gates.
FlyAI evolves the Shard inference framework. The [upstream repository](https://github.com/leyten/shard),
c0mpute integration history and dated experiments remain attributed in the documentation.

Contributors can register signed GPU offers without an operator allowlist. Execution requires a compatible
model cohort, measured resources, an executable calibration template, a live lease and verified warmup.
Placement prefers nearby or complete regional rings and expands when necessary. Co-located distinct GPUs
are allowed; shared host budgets and actual directed routes determine suitability.

The current implementation includes signed pipeline sessions, remote coordinator exclusion, persistent
resource ownership, exact calibrated heterogeneous placement, authenticated HTTP/SSE, globally bounded
multi-ring queues, request cancellation and original-prompt replay with prefix checking. Each ring has a
serial worker; separate rings execute concurrently. Optional private SQLite journals retain original requests
and committed prefixes for owner-fenced, same-cohort replay. This is not distributed coordinator HA.
Token-step scheduling is opt-in and requires an explicit backend contract. The distinct V4 native
tokenwise greedy cohort retains scalar state lanes and growing-prefix reuse. A new independent native
batch cohort runs 2–4 resident request states through actual equal-frontier batch forwards with bounded
socket pipelining and per-job receipts. The `/2` request-row math fixes the frozen B4 CPU
trajectory at zero tolerance. Optional partial attention/dense graphs use request-size/context buckets;
MoE and exact-width Indexer remain eager. This is not fused grouped GEMM or scalar whole capture.
Actual CUDA/FP4 correctness and performance remain pending; batching and its graphs default off.
See [current implementation and acceptance](docs/V4_DEEPINFER_IMPROVEMENTS.md).
Receipts bind signed activation commitments, job freshness and layer coverage; they do not
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
The [service-hardening guide](docs/V4_SERVICE_HARDENING.md) covers honest unknown SLO metrics,
local pinned replay auditors, durable recovery, SSE cursors and the four/six-card plus soak/CUDA release gate.
Strict production services are `engines/gpt_oss/network_service.py` and
`engines/deepseek_v4/v4_network_service.py`; they require full checkouts, pinned signing identities,
model content identities, calibrated local stage templates and configured inference routes. Explicit
legacy CLI modes retain older experiments. Public control/HTTP listeners require TLS, while existing
sidecars own peer transport and NAT handling.

The current selected regression spans **160 test files: 2548 passed, 11 skipped, 117 GPU deselected**, with stable source/test/tool hashes and one inherited Windows ACL subprocess encoding warning. The historical CPU/socket/HTTP regression passed **1813 tests with 8 environment-dependent skips**; 66 GPU-marked tests were deselected. The subsequent deployment/resource regression passed **218 tests with one POSIX case skipped on Windows**; the process-lifecycle suite passed **51 tests on WSL Ubuntu 22.04**. Overlapping results are not additive and do not claim a complete GitHub Python 3.12 CI or GPU performance pass.
Two additional CUDA cache/graph-pointer gates were conditionally skipped on that CPU-only runtime.
These are earlier-build records, not the current hardening regression total or GPU acceptance. Historical GPU numbers above retain their original workload,
recipe and timing scope. Actual native MXFP4 GPU execution and new V4 four-card >=40 / six-card >=30 tok/s
acceptance still require live cluster verification. Fixed-K adaptive depth is available; mixed-K experiments
charge their full greedy-control overhead and cannot use a single-shape prediction as a certified speed result.

## License

[Apache License 2.0](LICENSE) © 2026 leyten. Vendored references retain their own licenses and provenance.
