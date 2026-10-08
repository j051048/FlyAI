# 2026-10-08 六卡真机报告复核与复测方法

本页依据用户提供的调通总结与差距归因文本，对照仓库源码和历史收据。
这不是重新执行的 GPU 验收，也不修改历史原始 JSON。报告版本是 `83b5d16`
加现场 whole-graph 补丁和 Hadamard shim；未给出补丁 diff、冻结协议与原始签名收据。
后续比较应把该组合视为独立构建，不能只登记 Git HEAD。

## 能确认的进展与证据边界

- 报告称六个实际 RTX 5090 阶段完成原 V4-Flash 的 43 层推理，切分为
  `8/8/8/8/8/3`，尾部额外持有 embedding、输出参数和三个 MTP 块。
- 四个散文请求报告为 `27.43/24.55/23.73/24.36 tok/s`，中位数 **24.455**；
  单次最佳 **28.19**。如果另把 w0 定义为未完成预热的样本，后三次中位数是
  **24.36**。预热排除规则必须在执行前固定。
- 正确回答、完整计数和连贯文本是功能证据，不能代替同请求 greedy token/state
  对照、模型完整摘要或分布式收据。此次明确使用 legacy，不能登记 strict proof PASS。
- “峰值内存约 56GB < 150GB 磁盘”混用了资源维度。应分别记录目标 filesystem
  峰值占用、进程 RSS/cgroup memory peak 和 pinned RAM；30GB RAM 节点不能
  因磁盘有 150GB 就被判定能承担 56GB 主机内存峰值。

历史 [v4-flash-matrix-20260802.json](receipts/v4-flash-matrix-20260802.json)
写的是 **欧盟四国六台 distinct machines**，不是同国。headline 分别声明
peak 30.29、median 30.15；四项样本中位数是 30.151，去掉首项后的三个 warm
样本中位数是 30.178。“discard first”与标注值略有口径差异，应保留原记录并单列重算。
历史缺 exact prompt IDs、checkpoint 全摘要和可独立验的 raw receipts；
`receipts_ok: true` 是作者声明，不是新验证器得到的证据。

## 差距归因中需要纠正的推断

1. **尚不能排除代码或运行路径影响。** 不同 prompt、网络、长度和构建导致
   无法直接归因；现场修复了真实启动阻断，且 Hadamard fallback/graph 变化可能
   改变计算耗时。应进行同构建内 A/B 与固定 workload 的构建间比较。
2. **632ms/28ms 不是实测 RTT 分解。**
   [V4_PIPELINE_EFFICIENCY](V4_PIPELINE_EFFICIENCY.md) 的旧结果来自 throughput
   拟合，并明确记载当时没有测 RTT。不能据此给新 SSH 环划出 96% compute，
   更不能同时据此断言网络是最大瓶颈。
3. **g 不是统一的接受率。** serial 路径采用 `generated/rounds`，但当前
   pipelined 路径的 `cycles=cancels+1`，返回的 rounds 也是这个取消周期数。
   因此 256 token、零取消可以正常得到 `g=256`，它不受单个 block=8 的大小限制；
   没有 draft 的特殊路径也不能靠这项比值证明投机质量。要取原始 cycles/cancels、
   路径模式、accepted/drafted、accept histogram、
   已验证/作废帧计数核查。接受质量还受权重、tokenizer、draft 状态、模型和形状影响。
4. **cap/floor 不能凭名称补环境变量。** 当前流水线使用 `V4_SPEC_DEPTH`
   和 `V4_REFILL_FLOOR`；depth 默认 16、floor 默认 1。B=8、W=16 的
   `nblk=min(B,W-1)` 已可得到 9 个有效在途 frame 的上限。
   没显式配置不证明“半配方”；应先查看实际 parsed audit 与运行时占用。
5. **长度和计时分母必须同时对齐。** CLI tokPerSec 包含请求启动/prefill等
   时间，冻结 benchmark 另给 first/last commit 的 decode 口径。256→512
   可能降低固定启动成本占比，但也改变上下文、接受度和 capture 分支，不能预报增益。
6. **拓扑影响需要测量。** B/C/D/E 互不可达可以由路由探测证明；同宿主身份
   应另用 host inventory 确认，不能只从出口 IP 或相同容器私网 IP 推断。
   直连公网与 SSH 跳板应分别记录方向、caller-local 端点和真实路径。

## 先修可靠性，再做性能 A/B

| 优先级 | 工程项目 | 验收方式 |
| --- | --- | --- |
| P0 | native ModelArgs 与 tokenizer 加载解耦，保持配置文件/hash | native dtype=fp8 下本地 tokenizer 加载；冻结 rendered IDs 与 special tokens |
| P0 | 服务统一检查 Hadamard backend，记录 native/fallback | 首 token 前执行真实 device/shape probe；fallback 与参考数值比较 |
| P0 | 显式绑定 Attention/Indexer/Compressor 的缓存与 RoPE 别名 | cold graph capture 前校验；reset/缓存扩容/重绑后 state 对照；GPU 测试 |
| P0 | CPU selftest明确强制 CPU kernel backend | 有 CUDA 可见的机器也不能偷偷调用 tilelang CUDA kernel |
| P1 | 自动分片包含所有 tail/DSpark 依赖 | 新 [WEIGHT_PREPARATION](WEIGHT_PREPARATION.md) 的完整哈希与 Stage.load strict coverage |
| P1 | 统一环境注入、全环诊断与首因日志收集 | 每个进程报告 requested/parsed/effective/declined；EOF 定位 stage/peer/operation |
| P1 | 隧道先验证、tail-first 启动、最后实际签名 warmup | 单纯 LISTEN/TCP 成功不能作为 READY；保留失败原始 traceback |
| P2 | 同一 host GPU/CPU/NIC 竞争与端到端耗时分解 | 每段 forward/DMA/队列、传输编码/发送、回包等待与链路探测 P50/P95 |

图别名现场补丁还需要完整 diff。prefill 通常会经过 eager 绑定，因此仅凭总结
不能证明所有 whole-graph 冷启动都存在同一先后顺序；不能只补 None 掩盖错别名或失效 view。
Hadamard 的自逆误差也不是 CUDA bit-match 证明，fallback 会改变源码/内核身份并要求新校准。
不要用删除 native dtype、添加全局 torch.fp8 假属性来绕过 tokenizer 问题。

本轮权重准备工程已加入共享 [v4_tokenizer.py](../engines/deepseek_v4/v4_tokenizer.py)，
coord、gateway 和 benchmark 都直接使用本地 `PreTrainedTokenizerFast` 资产，
保留 native config 与其摘要；合法 HF 视图下的编码、特殊 token 和模板对照通过。
自动 stage 选择也已包含 DSpark tail 的 embedding 与完整 MTP 依赖。
后续 P0–P2 改造已经把 Hadamard 服务注册、whole-graph 别名校验/重捕获、
CPU selftest 隔离、strict 手动部署/隧道诊断和原始计数合同写入代码，见
[V4_OPERATIONS](V4_OPERATIONS.md)。实现依据是本仓库的统一合同，未合并未知的现场补丁；
GPU 数值、真实隧道和性能门槛仍需下一轮集群实测。

## 固定复测矩阵

先按 [V4_BENCHMARK](V4_BENCHMARK.md) 用 prepare/run/verify/compare 固定
checkpoint、全 catalogue、tokenizer、源码及脏补丁摘要、所有实际 V4 环境、硬件与端点。
公开性能报告采用完整 strict 部署计划和实际原始签名收据；legacy 结果独立保存。

每个单元报告原始 repetitions、中位数、P95、TTFT、decode/request tok/s、accepted/
proposed tokens、valid/stale frames、有效在途数及每段 compute/DMA/queue。
兼容字段 g 需另附 metric definition；明确区分 tokens per cancel cycle、
tokens per verified frame 和 accepted/proposed ratio，不把同名字段混作比较依据。
先完成所有可到达位置/宽度的 capture warmup，cold 的时间另列；不要执行后挑选丢弃规则。

| 实验 | 固定项 | 只改变的变量 |
| --- | --- | --- |
| 同条件工作量矩阵 | 原权重/构建、同环、同上下文、512 new token | Rust code、novel prose、math、copy 四类冻结 IDs |
| 长度矩阵 | 同 prompt IDs/构建/环/flags | 256 与 512 new token，分别报告两种时间分母 |
| 投机对照 | 同 prompt/环/上下文、预先验证数值 | greedy、DSpark B4/B8、depth8/16、floor1/2，记录真实有效在途数 |
| 传输对照 | 同阶段计算/权重/workload | 同一实际端点路径直连与可用隧道；不可达边不伪造直连结果 |
| 构建对照 | 同 frozen protocol、原始 patched source 两份完整归档 | baseline 与改造构建，先 token/state 对照，再比较耗时 |

原历史 Rust prompt 没有 exact IDs，当前只能做同类别的新基线，不能称精确复现原实验。
不应通过筛选高 g 文本替代真实应用 workload 的固定成绩。某单元未通过 tokenizer、
权重覆盖、数值或资源门控，停止该单元并保存首因证据，不能继续登记速度 PASS。

这些方法不要求删除旧 GPT-OSS 模型或终止正在服务的环。共享磁盘和 GPU 调度按实际
预算处理；无需再用“整份文件必须本机驻盘”的旧假设指导新的 V4 stage artifacts。
