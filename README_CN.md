# FlyAI / Shard

<div align="center">

[主 README](README.md) | **简体中文**

</div>

FlyAI 是基于上游 Shard 的分布式推理 fork。目标是允许任意贡献者登记 GPU，优先选择低延迟的近邻与同区域节点，不足时再扩区，将模型切成连续层块执行。同机多 GPU 是合法部署；是否加入执行环还要满足运行时兼容、真实资源、可达链路和租约要求。相同公网 IP 不能证明是同一物理机，也不能当作共享内存域。

**当前状态：2026-10-08，代码基线 `a6e96e3` 加本地权重准备、换环及 P0–P2 运行改造。** 本轮选定的本地非 GPU 回归集合为 **1620 通过、3 跳过、41 项 GPU 标记未选入**，不是全仓 CI，也没有完成新的 GPU 集群速度验收。当前指南、历史研究和上游来源见 [文档导航](docs/DOCUMENTATION_INDEX.md)。

## 当前能力与边界

| 能力 | 当前实现 | 验证范围与限制 |
|---|---|---|
| 开放节点登记 | 带签名的 offers、TTL、精确模型 cohort | 身份签名不是 GPU 硬件认证；缺测量节点可留在池中，不能直接执行 |
| 自动组环 | 区域优先、异构层段、实测路线与有限流水线预测 | 比较少量强卡和强弱混合的总成本；预测不是性能验收 |
| 节点资源占用 | SQLite prepare/commit/renew/release、fencing、共享 RAM/pin 预算 | 加载与空闲驻留也持有租约；旧任务和进程确认清理后才释放 |
| V4 混合运行时 | 本机 RAM 专家池、固定 GPU 缓存、按需 DMA、受控预取和 KV 工作集 | CPU 参考与专项测试已覆盖；实际 GPU 数值、命中性能和长期运行仍需验收 |
| V4 权重准备 | 有界流式转换、局部分片完整摘要、实际文件与准备峰值预算、原子发布及请求边界替换 | 需配置可信目录/来源与已校准候选；没有足够容量时明确拒绝，不清空运行模型 |
| V4 / GPT-OSS 服务 | 签名会话、多环路由、租户认证、队列、SSE、取消与前缀恢复 | 每环请求串行，多环可并行；job/队列不持久化，未实现服务 HA |
| GPT-OSS 下载与量化 | 固定 HF revision、完整摘要清单、转换前拦截反量化、转换后 packed 布局校验 | 布局校验不证明 native GPU 内核执行正确或吞吐达标 |
| 收据与本地仲裁 | 验证 signer、层覆盖、相邻 roots、job 和 nonce；独立重放挑战辅助 | 签名记录不是完整计算证明；本地辅助没有实际链上罚没 |

没有一个通用 backend 可以直接接入任意模型。MiniMax-M2.5、Kimi-K3、DeepSeek-V4 有独立引擎；GPT-OSS 使用现有 `phase0/specpipe.py` 并接入生产服务 adapter。GLM 等 `research/` 脚本属于历史实验，不能直接等同于当前生产入口。

严格 GPT-OSS 节点目前要在磁盘上保留完整下载清单列出的快照，再按分配层段加载 GPU；通用分块拉取路径不等于该入口已支持只下载本段。本轮没有实现 partial-inventory。

## 部署与运行实操

- **GPT-OSS：** [下载、部署与生产合同](docs/GPT_OSS_PRODUCTION.md)。严格入口要求完整 `ModelCohort`、验证过的实际权重、部署计划、节点 signer 与协调器 key。
- **V4：** [当前阶段与边界](docs/V4_NEXT_PHASE.md)、[集群部署指南](docs/V4_CLUSTER_DEPLOY_GUIDE.md) 和 [固定性能验收](docs/V4_BENCHMARK.md)。严格服务使用 `engines/deepseek_v4/v4_network_service.py`；旧 `v4_gateway.py` 是兼容接口。
- **V4 自检与排障：** [运行操作指南](docs/V4_OPERATIONS.md) 提供共享初始化、strict SSH 部署、隧道检查、健康诊断和单变量复测。
- **小磁盘节点：** [流式权重准备与请求边界换环](docs/WEIGHT_PREPARATION.md)。当前原 V4 native 适配支持局部分片；GPT-OSS 的完整下载 inventory 要求保持原合同。
- **开放网络：** [节点登记、资源租约、组环和服务](docs/OPEN_INFERENCE_NETWORK.md)。
- **M2.5 旧环：** [独立兼容运行指南](phase0/DEPLOY_M25.md)。其协议、依赖和启动器不会自动获得新 GPT-OSS/V4 会话合同。

默认开放网络数据链路使用 libp2p sidecar 的节点身份与加密连接，Python 只处理 JSON/原始张量帧。节点仍要授权预期环邻居；严格会话另绑定计划和协调器签名。原始 TCP 的 `SHARD_PSK` 属于明确兼容模式。公开 V4/GPT-OSS HTTP 服务必须启用 TLS 和租户认证；私钥与下载 token 留在本地受保护文件，不写入计划、收据或命令值。

## 数值正确性与隐私

验收必须固定 checkpoint、源码、量化布局、wire 模式、内核开关、context 和验证形状。同环贪心 token 一致、阶段状态一致、内核数值一致和不同后端的浮点位一致是不同结论；切换形状与量化内核后要重新对照。

计算节点仍能看到激活值。V4 可选 sealed-ID 模式只向 head、需要 hash 路由的 stage 和 tail 提供独立解密能力，隐藏普通中间节点的原始 IDs；它不隐藏激活值，也不防止受信任接收者或同一主机管理员泄漏数据。GPT-OSS 的会话身份认证不等于提示保密。边界见 [V4 信任说明](docs/V4_TRUST_BOUNDARIES.md) 和 [收据与证明范围](docs/PROOF.md)。

## 历史实机记录

这些数字保留原日期、配置和原始证据，不是当前 fork 的重新跑分或服务 SLA。

| 日期与记录 | 历史部署 | 报告速度 | 证据范围 |
|---|---|---|---|
| [2026-08-02 V4 矩阵](docs/receipts/v4-flash-matrix-20260802.json) | 6× RTX 5090，4 个欧洲国家；前五卡各 8 层，tail 3 层与 DSpark | 汇总 30.15 tok/s | 当时同环贪心 token 对照；不能推广为单机后端或所有浮点位一致 |
| [2026-06-19 GPT-OSS WAN](docs/receipts/gpt-oss-120b-wan-20260619.json) | 120B MXFP4，3×4090 的 12/12/12 层段与同区域协调器 | 约 40 tok/s | 历史运行记录；需分别核对自报数据、签名和独立复现范围 |
| [2026-06-18 GLM WAN](docs/receipts/glm52-nvfp4-wan-20260618.json) | 744B NVFP4，分散专业 GPU、CUDA Graph 草稿 | 约 30 tok/s | 历史研究入口，不能作为当前生产路径的验收 |

V4 **4×5090 ≥40、6×5090 ≥30 tok/s** 是新增路径的固定待验收目标。CPU 测试、模拟器与旧收据不能替代当前 GPU 集群对照。跨节点专家副本和 CPU 主推理不在当前 V4 服务范围。

## 仓库结构

```text
shard/       协议、资源合同、注册、规划、租约、会话、HTTP/SSE 和收据
engines/     minimax_m25、kimi_k3、deepseek_v4、gpt_oss 服务 adapter
phase0/      现有通用/GPT-OSS 运行时、下载、部署与基准工具
sidecar/     Go libp2p 身份、加密传输、NAT/relay 和内容路由
tests/       本地、参考与专项测试；GPU 测试有独立条件
research/    历史实验与原型
docs/        当前指南、历史研究与 receipts 档案
```

## 项目方向与上游来源

下一阶段按当前真实执行路径完成 GPU 数值与状态对照、短/长上下文性能矩阵、持续运行和故障恢复验收，再决定扩展模型与容量。通用 `ModelRuntime` 收敛仍在进行中，分布式训练与通用计算是后续方向。

上游 Shard 为 [c0mpute](https://c0mpute.ai) 的推理引擎；外部经济、支付与运营部署状态不能由本仓库代码或测试证明。上游 [技术报告](docs/paper/main.pdf) 与 [DOI](https://doi.org/10.5281/zenodo.21178430) 保留供研究引用。

## 开源协议

采用 [Apache License 2.0](LICENSE)，保留上游 © 2026 leyten 的版权与来源。
