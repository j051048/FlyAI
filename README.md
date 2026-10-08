# Shard

<details>
<summary>🌐 <b>点击在此处直接切换 / 展开【简体中文版】（原地阅读，无需跳转）</b></summary>

---

# Shard (中文版)

**无许可算力网络的底层引擎** —— *类似于算力世界的 BitTorrent，共享显存（VRAM）与计算能力而非磁盘空间。* 任何人都可以提供具有受支持运行时、通过资源准入检查的 GPU；网络将它们汇聚成分布式集群，运行远超单卡容量的超大模型。长期愿景是构建覆盖全球的去中心化算力织网（compute fabric）：承载更多、更大的模型，并最终扩展至分布式训练与通用计算。Shard 是连接这一切的底层协议。

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21178430.svg)](https://doi.org/10.5281/zenodo.21178430)

**技术报告：** [跨公网以交互式速度运行 229B MoE 模型的分片推理](docs/paper/main.pdf) —— 在横跨 5 个国家的消费级 GPU 上完成实测，所有基准测试数据均由 [`docs/receipts/`](docs/receipts/) 中带签名的收据（receipts）所背书。

**当下已验证的能力：分片推理（Sharded Inference）。** 将一个单张显卡无法容纳的超大模型切分为连续的层块（每个 GPU 承载一个分片），通过在公网上按序流式传输各分片的激活值（activations）来处理请求。模型层分布在多个节点，无需单个节点持有完整模型。每个请求由协调器驱动；协调器持久化故障切换仍待实现。

Shard 是 [c0mpute](https://c0mpute.ai) 的推理服务引擎。它由两部分组成：一个协议**骨架（Spine）** —— 包含底层通信协议（wire）、环形拓扑（ring）、签名凭据（receipts）与调度编排（placement）；以及针对各模型的独立优化引擎 —— 因为要在公网上实现交互式的推理速度，必须深入底层算子（kernel）进行定制优化。目前已提供三个引擎：**MiniMax-M2.5**（betanet 概念验证模型）、**Kimi-K3** 与 **DeepSeek-V4-Flash**（最新成果，详见下文）。长期架构将把所有引擎统一收敛在单个 `ModelRuntime` 接口后（参见 [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md)），使网络能运行任意模型；后续实测的 GLM-5.2 与 gpt-oss-120B 运行记录已证明该引擎能够从消费级显卡平滑扩展至前沿超大模型规模。

## DeepSeek-V4-Flash (284B)：运行于 4 个国家的 6 张消费级 RTX 5090

**历史汇总报告预热后 30.15 tok/s，投机输出与同环贪心基线的 token 一致。** 部署在 6 张*完全独立*的 RTX 5090 显卡上 —— 分布在波兰、捷克、丹麦与爱沙尼亚 —— 纯走公网连接，前五张卡各承载 8 层，末卡承载 3 层，无需共享物理机。投机解码（Speculative Decoding）运行在 DeepSeek 原生的 DSpark 草稿头上，其 3 个多标记预测（MTP）块连接最后 3 层，因此完全驻留在末端节点（tail box）上。

| 部署环境 | tok/s（预热后） | 输出精度与确定性 |
|-------|--------------|--------|
| DeepSeek-V4-Flash 284B（13B 激活参数）FP4，跨 4 个欧洲国家的 6× RTX 5090，公网环境，流水线 DSpark 投机解码 | **30.15** | 贪心采样，历史汇总报告同环 token 一致 |

连续三次预热运行的中位数，各次波动在 0.17 以内（30.18 / 30.29 / 30.12）。
上下文 × 负载测试矩阵（包含数学、代码、散文、智能体工作流；0–2k 上下文）见于
[`docs/receipts/v4-flash-matrix-20260802.json`](docs/receipts/v4-flash-matrix-20260802.json)：
**51/51 个单元测试均与同环拓扑上的贪心基线结果 bit-identical 一致，68/68 份签名收据验证通过，0 故障。**

这项工作中比具体速度数字更重要的两项核心技术结论，均记录在 [docs/V4_FLASH_ENGINE.md](docs/V4_FLASH_ENGINE.md) 中：
1. 系统的整体吞吐量本质上是*往返延迟期间的有效在途计算量*，任何为了填满流水线而盲目增加推测深度的杠杆都会以接近 1:1 的比例削弱接受率 —— 将流水线推满至 99.6% 负荷反而会导致吞吐量**减半**，因为新增的候选帧是基于后续未被接受的预测历史推算的；
2. 草稿块（draft block）的推测长度存在一个内在最优区间，只有深入测量中间值而非仅看端点时才能显现。

## GLM-5.2 (744B)：横跨全美 6 个州的 7 张分散专业 GPU，跑在公网之上

**参数量达 7440 亿的前沿超大模型，在分布于美国 6 个州的 7 张 Blackwell 架构 GPU 上跨广域网（WAN）以 ~30 tok/s 速度服务 —— 贪心采样，完全确定性。**
GLM-5.2（NVFP4 精度，78 层）被切分为**每个节点承载 13 层**，分摊在 6 张 RTX PRO 6000 上；没有单张显卡能装下它，而是由 6 张卡共同协作。每个节点**仅加载其所属的层块**。协调节点（coordinator）不保存任何模型中间层 —— 仅维护 token embedding / head 以及一个经由 CUDA Graph 优化的轻量级 GLM-4-9B 草稿模型用于提出候选 token，并由分布式 744B 模型进行验证。

| 部署环境 | tok/s（预热后） | 输出特性 |
|-------|--------------|--------|
| GLM-5.2 744B NVFP4，跨美国 6 个州的 6× RTX PRO 6000（内华达 · 德州 · 明尼苏达 · 密苏里 · 犹他 + 华盛顿协调节点），公网连接，流水线投机解码 + CUDA Graph 草稿加速 | **~30** | 贪心采样，确定性输出 |

每次运行都会生成一份**可验证收据（Verifiable Receipt）** —— 记录独立的 GPU UUID、公网 IP、地理区域、实测广域网单跳 RTT（22–75 ms）、输出 token 哈希值以及无损优化一致性检查。该次运行的收据文件位于：[`docs/receipts/glm52-nvfp4-wan-20260618.json`](docs/receipts/glm52-nvfp4-wan-20260618.json)（验证步骤详见 [docs/PROOF.md](docs/PROOF.md)）。

一句话总结其核心论证：规模高达 7440 亿的前沿模型（远超单卡承载极限），在跨越不同网络的异构物理机上运行 —— 每次遍历激活值都要跨越整个国家传输 —— 依然能够达到真正实用的生成速度。

## 性能优化历程与路径

单纯基于公网进行管道式解码受制于网络延迟：每个 token 都需要完整往返一次，吞吐仅约 1–2 tok/s，基本无法实际使用。从 1.8 跃升至 30 tok/s 是通过一系列严谨可度量的优化演进达成的：

| 阶段步骤 | tok/s | 关键改进点 |
|------|-------|--------------|
| 朴素 KV 缓存解码 | 1.87 | 受往返延迟限制的基线（每个往返生成 1 个 token） |
| + 深度草稿投机解码（GLM-4-9B），中继折返传输 | 1.99 | 单次流水线遍历可确认多个 token |
| + **环形直接返回（Ring Direct-Return）** | 2.94 | 尾节点一跳直达协调节点 —— 仅需 7 跳环路，无需 12 跳中继折返 |
| + **异步流水线（Async Pipelining）** | 16.6 | 并发重叠多个在途验证块 → 吞吐量受算力限制而非受延迟限制；WAN 耗时占比降至循环的 ~5% |
| + **CUDA Graph 捕获草稿头** | **~30** | 当网络延迟被掩盖后，草稿头占循环耗时的 94%；通过 CUDA Graph 加速（3.8×）彻底释放流水线吞吐 |

**核心洞见：在广域网（WAN）环境下，稀缺资源是往返时延（RTT）而非算力** —— 因此，在数据中心内价值微弱的投机解码在此处成为了决定性的关键。轻量草稿模型提出 K 个候选 token；分布式 744B 模型在单次流水线遍历中予以验证；贪心策略接受最长匹配前缀。随后产生了两项复合增益：

- **基于环形拓扑的异步流水线**：由于采用了直接返回机制，网络中可以同时并发多个在途验证块。协调节点持续推测生成并在不等待上一批次返回的情况下连续向流水线注入任务 —— 从而让整个循环以流水线的*吞吐率*而非*单次往返延迟*运行。曾经制约所有前人尝试的广域网延迟，在整体耗时中的占比缩减到了约 5%。
- **CUDA Graph 草稿头加速**：一旦网络延迟被掩蔽，GLM-4-9B 草稿模型（单 token 解码，受算子发射开销限制）便成为了整个循环 94% 的瓶颈。将其捕获为静态 CUDA Graph 后，单 token 生成延迟从 49.7ms 大幅降至 13.1ms（提升 3.8×）。其中最大的技术难点是让静态 KV 缓存在图捕获环境下依然支持投机回滚 —— 最终通过基于固定地址位置张量调度写入槽位解决；生成结果**在字节级别与原始 Eager 路径完全一致**，证明该项优化属于绝对无损优化（参见 `research/glm_swarm_nvfp4_cg.py`、`research/glm_swarm_nvfp4_cg_diff.py`）。

## 架构运作原理

Transformer 架构由若干层叠加而成。Shard 将层栈切分为连续的块，每个 GPU 分摊一个块。Token 的生成过程是通过按顺序将激活值穿透各个层块来实现的；每个节点仅为其负责的模型层维护局部 KV 缓存。

```text
    协调节点 (WA) ── GLM-4-9B 草稿头 (CUDA Graph) + Embed / LM_Head
         │
         ├─► stage0 ─► stage1 ─► stage2 ─► stage3 ─► stage4 ─► stage5 ─┐  （流水线并发验证块）
         │   NV         TX         (·)        MN         MO        UT    │
         │   0–12       13–25      26–38      39–51      52–64     65–77 │
         └──────────────── 直接返回 (尾节点直连协调节点，仅 1 跳) ───────┘
```

协调节点（入口节点）**不持有** 744B 模型的任何中间层 —— 仅维护草稿模型和一个轻量驱动器。每个轮次中：
1. 草稿头提出 K 个候选 token；
2. 协调节点将 `[cur, d₁..dₖ]` 发送至 stage 0（执行 Embedding）；
3. 分布式节点链在单次前向遍历中同时验证全部 K+1 个标记；
4. 尾部节点直接将各位置的 Argmax 计算结果一跳返回至协调节点（无需按原路反向中继折返）；
5. 协调节点按照贪心原则接受最长匹配前缀。

多个此类计算块在流水线中并发流动，草稿头利用静态 KV 缓存重放捕获的 CUDA Graph。贪心匹配是默认行为；该路径还原生支持**无损的 Temperature / Top-p / Top-k 采样** —— 由末端节点执行投机采样拒绝判断，从而使提交的 token 分布与目标模型完全一致，且相比贪心算法没有任何速度损失（参见 [`phase0/specsample.py`](phase0/specsample.py)）。

## 核心难点与工程挑战

在局域网同机房的 GPU 间切分模型已经非常成熟。但要跨越公网在异构物理机之间切分并达到交互级可用速度，是一项截然不同的挑战 —— 这正是 Shard 的核心价值所在。

- **延迟（Latency）：** 每个 token 都需要穿越完整流水线。投机解码将一次往返时延分摊至多个被接受的 token 上；异步流水线重叠多个在途前向传播，彻底打破了广域网延迟下限；CUDA Graph 保证本地算子开销微乎其微。
- **传输层可靠性（Transport）：** 激活值张量在每一步中都需要跨越公网传输。Shard 封装了专门的传输层 —— 具备毫秒级快速失败与自动重连机制、单链路健康监控，杜绝黑盒式的 "broken pipe"。通信协议基于无 pickle 依赖的帧结构（`phase0/wire.py`），通过共享密钥 `SHARD_PSK` 使用 ChaCha20-Poly1305 进行全量认证加密；被动监听者无法获取任何有效数据，伪造帧只会触发解析错误而绝无任意代码执行风险。（家用路由器的 NAT 打洞与中继回退正在 Phase 1 中推进；当前采用直连开放端口支持公网互联）。

## 设计原则

Shard 作为 c0mpute 的基础设施，坚守三项核心承诺：

- **无审查（Uncensored）：** 引擎原汁原味地运行模型本身，推理路径中不插入任何额外的内容审查过滤层。
- **去中心化（Decentralized）：** 任何人只需一条命令即可接入 GPU 节点并分配得到特定的模型层块。计算由多个节点协作完成，每个请求仍由协调器驱动。
- **隐私保护（Private）：** 没有单个节点能够持有完整模型 —— 这是一个良好的开端，但并非终点。传输线缆完全密封（经认证的加密传输，无 pickle 隐患），因此传输链路不存在泄漏；但*参与运算的节点*必须解密后才能计算其所属层，因此能看到经由它的中间激活值。恶意节点仍有可能从局部激活值中逆向推测出用户的少量 token。我们的应对方案 —— 将易泄露的边界层固定在受信任节点上、按请求动态安全路由、绝不过度宣称安全性 —— 详细规划见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。这是团队视作头等优先级并全力攻克的核心课题。

## gpt-oss-120B 在公网上达到 ~40 tok/s —— 消费级显卡实证

1200 亿参数模型（MXFP4 精度，36 层）分布在**跨越美国不同州的 3 张消费级 RTX 4090** 与 1 个协调节点上，达到 **~40 tok/s（峰值 ~42 tok/s），贪心采样，精确匹配**。这一成果验证了 Phase 3+ 无许可网络技术栈在普通 24GB 消费级显卡上的可行性 —— 即真实社区志愿者所拥有的硬件设备。当前的 **betanet 基础模型为 MiniMax-M2.5**（229B-A10B MoE），基于 libp2p 协议完成了热机验证与带签名收据校验（支持工具调用、多轮对话、长上下文）；gpt-oss-120B 与 GLM-5.2 则分别作为消费级显卡与前沿超大规模的模型扩展性标杆。

该次运行的可验证凭据见：[`docs/receipts/gpt-oss-120b-wan-20260619.json`](docs/receipts/gpt-oss-120b-wan-20260619.json)。

从受延迟制约的 ~18 tok/s 逐步攀升的实测演进：

| 阶段步骤 | tok/s | 关键改进点 |
|------|-------|--------------|
| 流水线投机解码（4-stage） | 25.8 | 异步草稿重叠 + 多个验证块在途 + RTT 最优环路排列 |
| + **3-stage（12 层）环路** | 28.8 | 更饱满的 stage → 4 次 WAN 跳步替代 5 次（12 层恰好适配 24GB 卡） |
| + **协调节点就近同区域部署** | **~40（峰值 ~42）** | 协调节点无需加载模型权重层，可置于任意位置；将其移出跨国超长链路，单环延迟从 174ms 锐减至 102ms |

最后一步优化往往最容易被忽视：系统中资源消耗最小的节点 —— 完全不持有模型层的协调节点 —— 如果与集群相隔一个大洲，每生成一个 token 就会凭空付出两次长途跨洲往返。将其迁移至集群节点相近的区域，即可零成本换取约 40% 的端到端延迟缩减。完整研究报告见：[docs/research/wan-speculative-decoding.md](docs/research/wan-speculative-decoding.md)。

## 仓库结构

```text
shard/       协议骨架（Spine），与具体模型无关：
             node.py（ModelRuntime 接口）、transport（传输）、scheduler（调度）、
             topology（拓扑）、manifest（配置清单）、fetch（拉取）、
             receipt（签名凭据）、challenge（挑战校验）
phase0/      模型专属引擎与部署工具：
               m25_*.py  MiniMax-M2.5   — betanet 服务路径
               k3_*.py   Kimi-K3
               v4_*.py   DeepSeek-V4-Flash
             以及 wire.py（安全封包）、mesh.py（边缘 RTT 测量）、运行与基准测试工具，
             以及引入的不可篡改参考实现目录 (*_ref/)
research/    研究实验与原型 —— GLM-5.2 集群驱动 (glm_swarm_nvfp4_*)、V4 性能分析探针等
docs/        ARCHITECTURE、ROADMAP、MODEL_RUNTIME、NETWORK、INTEGRATION、PROOF.md、
             V4_FLASH_ENGINE.md、receipts/ 收据归档以及各项研究记录
```

各模型引擎允许根据自身架构深度定制优化算子，但协议骨架（Spine）必须保持通用与统一。`tests/test_engine_boundaries.py` 自动化测试严格保障分层边界：任何引擎不得相互交叉引用，`shard/` 也不得反向依赖任何特定引擎。

## 项目路线图

- **Phase 0 —— 传输协议验证（已完成）：** 实现可靠的多阶段跨机模型切分与服务承载。
- **Phase 1 —— 广域网环境适配：** 支持 NAT 内网穿透、中继回退、激活值量化压缩与边际链路监控。
- **Phase 2 —— 投机解码加速：** 在集群上实现草稿与验证分离 —— **已在 GLM-5.2 744B 上达成公网贪心 ~30 tok/s**（以及 gpt-oss-120B 的 ~18–25 tok/s）。现已支持**无损的 Temperature / Top-p / Top-k 采样**（[`phase0/specsample.py`](phase0/specsample.py)），实测收据：[docs/receipts/sampling-lossless-20260623.json](docs/receipts/sampling-lossless-20260623.json)。
- **Phase 3 —— 无许可弹性集群：** 单行命令入网、跨异构 GPU 的动态层切分分配、按 Token 结算收益、故障容错与自愈 —— **已验证请求过程中途节点故障自愈**（生成中途剔除节点，请求自动在备用节点恢复并完成；`phase0/heal.py`，[收据](docs/receipts/fault-tolerance-20260623.json)）。
- **DeepSeek-V4 双资源与生产加固（实际执行路径已接入，GPU 集群验收待完成）：**
  1. 固定槽位 RAM 专家池、按需 DMA、受控预取与签名指标已接入，GPU 的命中/未命中性能需要实测。
  2. 可选逐层 KV 工作集保留原压缩器数学，将压缩历史放主机内存；显式 GPU/host 配额也包含回滚预算，支持的上下文受工作区容量约束。
  3. 预填充仅分块 Attention/Indexer 查询，保留原投影、Compressor 与 MoE 形状；首次实际形状需通过输出与状态的数值门控，首次峰值仍需完整参考路径。
  4. 密封 token IDs、身份绑定收据、认证串行服务、租户公平队列、取消/期限和原请求重放恢复已实现；激活值仍可见，服务尚不具备持续批处理与持久化 HA。
  5. 同机多 GPU 准入检查共享 RAM/锁页预算与同时 H2D 带宽。历史跑分不能认证新增路径；4×5090 ≥40、6×5090 ≥30 tok/s 仍为待验证目标。
  部署与限制见 [docs/V4_NEXT_PHASE.md](docs/V4_NEXT_PHASE.md)，固定速度验收见 [docs/V4_BENCHMARK.md](docs/V4_BENCHMARK.md)。跨节点专家副本与 CPU 主推理路径不在当前 V4 服务范围。
- **统一引擎架构（进行中）：** 将所有服务路径抽象收敛于统一的 `ModelRuntime` 接口（[`shard/node.py`](shard/node.py)），使网络能运行*任意*开源模型；模型层接入生态标准，核心壁垒（环拓扑、高效传输、投机验证）保持自研。规划见 [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md)。
- **远景目标 —— 超越推理：** 利用相同的无许可基础底座（节点身份、安全传输、内容寻址权重分发、去中心化验证与结算通道）承载通用算力与分布式大模型训练。

完整设计与执行规格见：[docs/V4_HYBRID_RUNTIME.md](docs/V4_HYBRID_RUNTIME.md) 与 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源协议 © 2026 leyten

---

</details>

[English](README.md) | [独立中文页面 (README_CN.md)](README_CN.md)

**The engine for a permissionless compute network** — *BitTorrent, but you share
VRAM and compute instead of disk.* Anyone can offer a GPU with a supported runtime and measured resource admission; the network
pools them into swarms that run models far larger than any single card holds. The
long arc is a worldwide compute fabric: many models, ever bigger, and eventually
training and general compute. Shard is the protocol that connects it.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21178430.svg)](https://doi.org/10.5281/zenodo.21178430)

**Technical report:** [Sharded Inference of a 229B-Parameter MoE over the Public Internet at Interactive Speed](docs/paper/main.pdf) — measured on five countries' consumer GPUs, every number backed by a signed receipt in [`docs/receipts/`](docs/receipts/).

**Proven today: sharded inference.** A model too large for any single card is split
into contiguous blocks of layers — one shard per GPU — and a request is served by
streaming activations through the shards in order, over the open internet. No
single host holding the full model is required. A coordinator drives each request; durable coordinator failover remains future work.

Shard is the serving engine for [c0mpute](https://c0mpute.ai). It is a protocol **spine**
— wire, ring, receipts, placement — plus one engine per model, because getting a model to
interactive speed over the open internet means tuning it down to the kernels. Three engines
exist today: **MiniMax-M2.5** (the betanet proof-of-concept), **Kimi-K3**, and
**DeepSeek-V4-Flash** (newest, below). The long-run direction is to drive all of them behind a
single `ModelRuntime` interface ([docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md)) so the network
runs any model; the GLM-5.2 and gpt-oss-120B runs below proved the engine scales from consumer
cards to frontier size.

## DeepSeek-V4-Flash (284B) on six consumer RTX 5090s in four countries

**Historical summary: 30.15 warm tok/s, with speculative tokens matching same-ring greedy controls.** Six *distinct*
RTX 5090s — Poland, Czechia, Denmark, Estonia — connected over the public internet, 8 layers
on each of the first five cards and three on the tail, with distinct hosts. Speculative decoding runs on DeepSeek's own DSpark
drafter, whose three MTP blocks tap the last three layers and therefore live entirely on the
tail box.

| Setup | tok/s (warm) | Output |
|-------|--------------|--------|
| DeepSeek-V4-Flash 284B (13B active) FP4, 6× RTX 5090 across 4 EU countries, WAN, pipelined DSpark speculation | **30.15** | greedy, historical same-ring token parity |

Median of three consecutive warm runs within 0.17 of each other (30.18 / 30.29 / 30.12).
A context × workload matrix (math, code, prose, agentic; 0–2k context) is in
[`docs/receipts/v4-flash-matrix-20260802.json`](docs/receipts/v4-flash-matrix-20260802.json):
**51/51 cells bit-identical to a greedy baseline on the same ring, 68/68 signed receipts, 0 faults.**

Two results from that work are worth more than the number, and both are in
[docs/V4_FLASH_ENGINE.md](docs/V4_FLASH_ENGINE.md). Throughput is *useful in-flight work over
round-trip latency*, and every lever that fills the pipe costs acceptance at roughly 1:1 — forcing
the pipe to 99.6% of its cap **halves** throughput, because the added frames speculate on a history
the ring then does not take. And the draft block has an interior optimum that only appears if you
measure the middle rather than the endpoints.

## GLM-5.2 (744B) across seven scattered prosumer GPUs, over the open internet

**A 744-billion-parameter frontier model, served at ~30 tok/s across seven
prosumer Blackwell GPUs in six US states — over WAN, greedy, deterministic.**
GLM-5.2 (NVFP4, 78 layers) is split **13 layers per node** across 6× RTX PRO 6000;
no single card holds it, six do. Each node loads **only its own block**. A coordinator
holds no model layers — just the token embedding/head and a small CUDA-graphed
GLM-4-9B draft that proposes tokens, which the distributed 744B verifies.

| Setup | tok/s (warm) | Output |
|-------|--------------|--------|
| GLM-5.2 744B NVFP4, 6× RTX PRO 6000 across 6 US states (NV · TX · MN · MO · UT + WA coord), WAN, pipelined spec-decode + CUDA-graphed draft | **~30** | greedy, deterministic |

Every run emits a **verifiable receipt** — distinct GPU UUIDs / public IPs / regions,
measured WAN edge RTTs (22–75 ms), the output token hash, and a lossless-optimization
check. This run's receipt: [`docs/receipts/glm52-nvfp4-wan-20260618.json`](docs/receipts/glm52-nvfp4-wan-20260618.json)
(see [docs/PROOF.md](docs/PROOF.md) for how a skeptic checks it).

That is the whole thesis in one line: a frontier-size model, far too big for any
single card, served across machines on different networks — activations crossing
the country on every traversal — at a speed that is actually usable.

## How it got there

Plain pipeline decode over WAN is latency-bound: one round-trip per token, ~1–2
tok/s, unusable. The path to 30 was a sequence of measured steps, each committed:

| Step | tok/s | What changed |
|------|-------|--------------|
| plain KV decode | 1.87 | latency-bound baseline (one token per round-trip) |
| + deep-draft spec-decode (GLM-4-9B), relay-back | 1.99 | one traversal commits several tokens |
| + **ring direct-return** | 2.94 | tail returns to the coordinator in one hop — 7 ring hops, not a 12-hop relay-back |
| + **async pipelining** | 16.6 | overlap many verify traversals in flight → throughput-bound, not latency-bound; the WAN drops to ~5% of the loop |
| + **CUDA-graphed draft** | **~30** | with the WAN hidden, the draft was 94% of the loop; CUDA-graphing it (3.8×) lifts the whole pipeline |

**The key insight: over WAN the round-trip is the scarce resource, not compute** —
so speculative decoding, marginal in a datacenter, becomes the whole game. A small
draft proposes K tokens; the distributed 744B verifies them in a single pipeline
traversal; greedy acceptance commits the verified prefix. Then two compounding wins:

- **Async pipelining over the ring.** Because the ring is direct-return, multiple
  verify chunks can be in flight at once. The coordinator drafts a continuous stream
  and pumps overlapping chunks into the pipeline without waiting — so the loop runs at
  the pipeline's *throughput*, not its *latency*. The WAN, which dominated every prior
  attempt, drops to ~5% of the loop.

- **CUDA-graphed draft.** Once the WAN is hidden, the GLM-4-9B draft (single-token
  decode, launch-overhead-bound) becomes 94% of the loop. Capturing it as a CUDA graph
  cuts it 3.8× (49.7→13.1 ms/tok). The hard part was making the static KV cache honor
  speculative rollback under graph capture — solved by driving the write slot through a
  static-address position tensor; the result is **byte-identical to the eager path**, so
  the optimization is provably lossless. (`research/glm_swarm_nvfp4_cg.py`,
  `research/glm_swarm_nvfp4_cg_diff.py`.)

## How it works

A transformer is a stack of layers. Shard splits the stack into contiguous blocks,
one block per GPU. A token is produced by passing activations through the blocks in
order; each node keeps a KV-cache for its own layers.

    coordinator (WA) ── GLM-4-9B draft (CUDA-graphed) + embed / lm_head
         │
         ├─► stage0 ─► stage1 ─► stage2 ─► stage3 ─► stage4 ─► stage5 ─┐  (verify chunks, pipelined)
         │   NV         TX         (·)        MN         MO        UT    │
         │   0–12       13–25      26–38      39–51      52–64     65–77 │
         └──────────────── direct return (tail → coordinator, 1 hop) ────┘

The coordinator (entry node) holds **no** 744B layers — only the draft and a thin
driver. Each round: the draft proposes K tokens; the coordinator ships `[cur, d₁..dₖ]`
into stage 0, which embeds them; the chain verifies all K+1 in one forward traversal;
the tail returns the argmaxes straight to the coordinator (one hop, not relayed back);
the coordinator greedy-accepts the longest matching prefix. Many such chunks are in
flight at once (the pipeline), and the draft replays a captured CUDA graph against a
static KV cache. Greedy is the default; the same path also does **lossless temperature/top-p
sampling** — the tail runs speculative-sampling rejection so the committed token distribution
exactly matches the target's, at no speed cost vs greedy (`shard/specsample.py`).

## Why this is hard

Splitting a model across co-located GPUs is well understood. Doing it across machines
on the open internet, fast enough to be usable, is not — and that is the part Shard
owns.

- **Latency.** Every token traverses the whole pipeline. Speculative decoding amortizes
  one round-trip over many committed tokens; pipelining overlaps the traversals so the
  WAN stops being the floor; the CUDA-graphed draft keeps what's left cheap.
- **Transport.** The activation tensor crosses the public internet on every step. Shard
  owns this layer — supervised edges that fail fast and reconnect, per-edge health
  logging, no opaque "broken pipe." The wire is authenticated and encrypted with
  pickle-free framing (`phase0/wire.py`; ChaCha20-Poly1305 under a shared `SHARD_PSK`),
  so a passive observer learns nothing and a forged frame is a parse error, not code
  execution. (NAT hole-punching + relay fallback for home routers is the remaining
  Phase 1 work; a direct open port stands in today.)

## Design principles

Shard is c0mpute infrastructure, held to its three guarantees:

- **Uncensored.** The engine runs models as-is. No content filter in the inference path.
- **Decentralized.** Anyone can join a GPU with one command and be assigned a block of
  layers. Distributed execution is driven by a per-request coordinator.
- **Private.** No node holds the whole model — a real start, not the whole story. The
  wire is sealed (authenticated encryption, pickle-free), so the leak is not on the
  path; but a *participating* node must decrypt to run its layer, so it sees the
  activations it processes. Intermediate activations can still leak a fraction of a
  user's tokens to a malicious node. The plan — pin leaky boundary layers to trusted
  nodes, per-request trusted routing, never overclaim — is in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It is the number-one open problem and is
  treated as one.

## gpt-oss-120B at ~40 tok/s over WAN — proving the engine on consumer cards

120B (MXFP4, 36 layers) across **3 scattered RTX 4090s in different US states** + a
coordinator, **~40 tok/s (peak ~42), greedy, exact**. This proved the permissionless
stack (Phase 3+) on plain 24GB consumer cards — the hardware a real volunteer runs. The
current **betanet base model is MiniMax-M2.5** (229B-A10B MoE), warm-validated over libp2p
with signed receipts (tool-calling, multi-turn, long context); gpt-oss-120B and GLM-5.2
remain the consumer-card and frontier-size scaling proofs.
This run's verifiable receipt (distinct GPU UUIDs / IPs / states, WAN edge RTTs, output
hash, sync-vs-pipelined token match): [`docs/receipts/gpt-oss-120b-wan-20260619.json`](docs/receipts/gpt-oss-120b-wan-20260619.json).

The climb from a latency-bound ~18 tok/s, each step measured:

| Step | tok/s | What changed |
|------|-------|--------------|
| pipelined spec-decode (4-stage) | 25.8 | async-draft overlap + many verify chunks in flight + RTT-optimal ring order |
| + **3-stage (12-layer) ring** | 28.8 | fatter stages → 4 WAN hops instead of 5 (12 layers fits a 24GB card) |
| + **coordinator placed in-region** | **~40 (peak ~42)** | the coordinator holds no model layers, so it can live anywhere; moving it off the cross-country leg cut the ring 174→102 ms |

The last step is the one nobody looks for: the cheapest node in the system — the
layer-less coordinator — was sitting a continent away from the swarm, paying two long
round-trips on every token. Putting it next to the stages, on the same scattered nodes,
was a ~40% latency cut for free. Full record:
[docs/research/wan-speculative-decoding.md](docs/research/wan-speculative-decoding.md).

GLM-5.2 (above) remains the **frontier-size** flagship — 6× the parameters at 744B;
gpt-oss-120B is the faster, consumer-card build target the network is bootstrapped on.

## Repository layout

    shard/    the protocol SPINE, model-agnostic: node.py (the ModelRuntime interface),
              transport, scheduler, topology, manifest, fetch, receipt, challenge
    phase0/   the engines, one per model, plus deploy tooling. Module prefix is the
              engine boundary:
                m25_*.py  MiniMax-M2.5   — the betanet serve path
                k3_*.py   Kimi-K3
                v4_*.py   DeepSeek-V4-Flash
              alongside wire.py (sealed framing), mesh.py (edge RTTs), launch + bench
              tooling, and the vendored reference trees (*_ref/), kept byte-identical
              and driven, never reimplemented
    research/ experiments — the GLM-5.2 swarm drivers (glm_swarm_nvfp4_*), the V4
              profilers, and the M2.5 probes that fed the proven path
    docs/     ARCHITECTURE, ROADMAP, MODEL_RUNTIME, NETWORK, INTEGRATION, PROOF.md,
              V4_FLASH_ENGINE.md, receipts/, and the research records

Engines are meant to diverge — each is tuned to its model down to the kernels — but the
spine must not. `tests/test_engine_boundaries.py` enforces both rules mechanically: no engine
may import another engine, and `shard/` may not import any engine. It also carries a named,
shrink-only list of the places that rule is still broken, so the debt is visible rather than
assumed. Flattening this into `engines/<model>/` + `vendor/` is planned; the boundary is the
part that matters, and it is already a test rather than a convention.

## Roadmap

- **Phase 0 — Transport, proven.** Reliable serving through a multi-stage split.
- **Phase 1 — WAN.** Different networks behind NAT: hole-punching, relay fallback,
  activation quantization, edge supervision.
- **Phase 2 — Speculative decoding.** Draft-and-verify over the swarm — **done at
  GLM-5.2 744B scale, ~30 tok/s greedy over WAN** (and gpt-oss-120B at ~18–25, above).
  Now **lossless temperature/top-p/top-k sampling** too (not just greedy): speculative-sampling
  rejection at the tail (`shard/specsample.py`), the committed distribution provably equal to the
  target's. Receipt: [docs/receipts/sampling-lossless-20260623.json](docs/receipts/sampling-lossless-20260623.json).
- **Phase 3 — Permissionless swarm.** One-command join, dynamic layer allocation
  across heterogeneous GPUs, per-token payouts, fault tolerance — **mid-request heal demonstrated**
  (kill a node mid-generation, the request resumes on a spare and completes; `phase0/heal.py`,
  [receipt](docs/receipts/fault-tolerance-20260623.json)).
- **DeepSeek-V4 dual-resource production hardening (integrated opt-in paths; GPU acceptance pending):**
  1. Local pinned expert pools, fixed GPU slots, demand DMA, bounded prediction and signed observations are integrated.
  2. Layer-local KV working sets preserve Compressor semantics while keeping compressed histories on the host. Explicit tier quotas include rollback; supported context is bounded by workspace capacity.
  3. Prefill chunks Attention/Indexer queries only. Projection, Compressor and MoE shapes are preserved; the first real shape must pass output/state parity and still pays a full reference peak.
  4. Sealed token IDs, identity-bound receipts, an authenticated serial gateway, fair tenant queues, cancellation/deadlines and original-request replay recovery are implemented. Activations remain visible; continuous batching and durable HA are not provided by this V4 gateway.
  5. Co-located GPU admission checks shared host RAM/pinning and simultaneous H2D I/O. Historical runs do not certify these new paths: four-card >=40 and six-card >=30 tok/s remain hardware gates.
  See [docs/V4_NEXT_PHASE.md](docs/V4_NEXT_PHASE.md) and [docs/V4_BENCHMARK.md](docs/V4_BENCHMARK.md). Cross-node expert replicas and a CPU main inference path are outside the current V4 service.
- **One engine, every model.** The serve path is being generalized behind a single
  `ModelRuntime` interface (`shard/node.py`) so the network runs *any* model, not one
  hand-ported architecture — the model layer is inherited from the ecosystem; the moat
  (ring, transport, spec-decode, verification) stays in-house. Direction in
  [docs/MODEL_RUNTIME.md](docs/MODEL_RUNTIME.md); in progress.
- **Horizon — beyond inference.** The same permissionless rails (identity, transport,
  content-addressed weights, verification, payment) are meant to carry general compute,
  training included. That is a separate execution core — honestly years past the inference
  fabric — named here as the direction, not a claim.

## 🚀 Cluster Deployment Runbook

For a complete, step-by-step operational guide on renting machines (e.g., **Vast.ai** / **AutoDL**) or setting up **self-hosted GPU rigs** to serve DeepSeek-V4-Flash:
👉 **[DeepSeek-V4 Cluster Deployment & Runbook (docs/V4_CLUSTER_DEPLOY_GUIDE.md)](docs/V4_CLUSTER_DEPLOY_GUIDE.md)**
- OS & CUDA recommendations, Python venv, PyTorch, and TileLang installation;
- Single-box 4-GPU & multi-node WAN/LAN pipeline topologies and per-stage CLI startup commands;
- Environment variable profiles for dual-resource hybrid RAM expert cache;
- Pipeline health checks and Phase 0 hardware acceptance validation.

Full design & execution spec: [docs/V4_HYBRID_RUNTIME.md](docs/V4_HYBRID_RUNTIME.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## License

[Apache License 2.0](LICENSE) © 2026 leyten
