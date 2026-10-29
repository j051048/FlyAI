# 六项执行与调度优化

更新：2026-10-29。实现依据是当前 FlyAI 与 Strata 主线 `46c152a1310f77cd2f0ce2c7cd213962df14fe83` 的工程分析。新路径均需要显式选择；本轮本地结果不能作为 GPU 吞吐或质量成绩。执行步骤见[六项真机验收](V4_SIX_ACCEPTANCE.md)，前轮专家分层、完整 MTP 图和草稿训练见[前轮说明](V4_DUALGPU_IMPROVEMENTS.md)。

## 1. 全环成本和投机

启用已校准的 RoundPolicy 后，reset 固定本次 trace 与 stage 数量，各 stage 在实际 step 前后采集专家缓存 DMA 计数增量、dispatch wall 和捕获状态。exclusive 的保全 D2H 与换入 H2D 均计入，CPU 模拟不能计硬件 DMA；这些是完成保全/提交复制的程序计数，不是 PCIe 硬件计数器。尾回复按 trace、帧 epoch、位置和 stage 顺序校验完整链；缺少任一段仍为 unknown。dispatch wall 不能当作 GPU 时间，各段重叠区间不能直接相加。

`exploration` 可配置 `every_windows`、`max_probes`、`stale_windows`，用于重新测量已过时的窗口代价；首次捕获和 fallback 状态变化不学习为稳态成本。现有加载草稿宽度、接受规则、回滚深度保留。当前控制器仍选择未发送候选前缀；不声称已经按选择宽度减少全部 MTP 草稿计算，也不动态改变数学执行路径。

## 2. 同机实际字节与收据

私有 host transport 配置可增加 `"digest_backend":"device_sha256"`，默认仍为 CPU。设备路径读取实际 GPU tensor 的标准 SHA-256，保持 plain 与 sealed、BF16 与 FP8 的原始字节顺序，收据仍按每帧 SHA-256 构造同一条根。网络上的摘要声明不能构造本地预哈希对象。

在 IPC 成功选用时，发送端不构造完整隐藏张量 CPU witness，接收端对实际目的 tensor 再计算摘要。此模式启动前编译 CUDA 扩展并运行填充分界已知答案检查；IPC 不可用时不消费未填充的共享内存。不匹配通道或槽位用尽仍可走原 socket；须记录实际回退。标准顺序 SHA 内核可能比 CPU OpenSSL 慢，只有完整交接 A/B 证明确有收益才能推荐。临时 tensor、摘要拼接、静态池和 CUDA driver 分配都计入预算。

## 3. 生产请求选环

network service 的可选 `production_routing` 接入真实 RingPool 请求路由。按上下文/输出长度、模式及并发条件使用已有实测形状桶，并保留不确定度和有限的新请求探索预算。候选必须是同 cohort、已 READY 且满足地域/租约约束的环。

精确 benchmark 的经济重组条件不放宽；这项实现不会自动生成任意新层分配，也不会将同公网 IP 视为共享主机。实际收益需用长短混合请求和相同到达时间测量，包括冷启动、队列、抖动和故障成本。

## 4. 持久状态和异长批量

`V4_NATIVE_BATCH_STORAGE=persistent` 选择独立 `v4-native-persistent-ragged-greedy/3` 合同。每请求持有稳定状态槽位，Q/KV/O、HC 和 MoE 进入一次真实 request-batch Stage.forward；原生 top-K、Compressor 和 sparse attention 按各行独立位置读取对应槽位。每步事务只备份可能写入的 ring 行、压缩行和固定递归状态，不复制整份 KV。

失败时恢复写集，无法恢复则隔离故障 owner；取消和复用槽位不能污染其他请求。当前 /3 要求 graph-off、驻留 KV/专家、无加载 MTP。原 /2 的 equal-frontier 与 partial graph 仍是独立路径，不把新模式报为已支持 ragged CUDA graph。

## 5. RAM 专家、分段图和就绪调度

`V4_EXPERT_SEGMENT_GRAPH_MIB` 是 Stage 主层的总图预算，按层分配；`V4_EXPERT_SEGMENT_GRAPH_MAX` 限制每层图数量。gate/shared/fixed physical slot 可以进入有界无状态图池，主机缓存调度留在图外；物理槽位换入不同专家后回放必须读取新权重，模型/地址/epoch 不匹配拒绝复用。

exclusive 只在显式分段预算、whole 的 eager-MoE seam、`V4_MOE_IN_GRAPH=0` 与驻留 KV 条件下扩展原图互斥边界。捕获失败恢复所有被触碰的 KV；恢复失败不能继续 eager。资源与实际执行观察进入静态配置/动态观察各自的字段。

`V4_EXPERT_READY_FIRST=1` 先提交已就绪热专家，再逐个接入冷专家，输出仍按原逻辑专家顺序折叠。`V4_EXPERT_READY_MIB` 限制暂存输出；超限采用原序流式路径。它不与预先占满槽位的 prefill producer 同时报告生效。CPU 仍是显式旁路，没有跨节点取专家。

## 6. 增量前缀和精确 KV gather

增量复用需要 operator-local 完整资格工件：实际 strict 环签名 canary、逐 stage 原生 GPU case、源码/模型/运行配置/执行模式身份，六类状态与隔离案例齐全。加载器逐签名复核并拒绝 pending 或手填 `gpu_verified=true`。首次请求和周期 canary 仍做完整 prefill 对照；任何不一致永久关闭当前策略，保留完整原路径。资格只覆盖 producer 实际执行的驻留 scalar greedy 边界，不覆盖 MTP 或其他数学合同。

通过资格后，已有签名 parent restore 加顺序 suffix 执行直接形成新前缀状态与首 token，省去当次完整 prefill；收据仍只记录真实新 forward，保留 parent restore 绑定。EOS、容量不足或资格失配回退原路径，租户/nonce/lease/源身份不放宽。

`V4_KV_INDEXED_GATHER=1` 要求 layer KV 和原生完整 prefill。ratio-4 的 Indexer 历史驻留设备，保留原 top-K 与顺序；decode 只搬入所选冷 compressed rows，ring 与新完成行在设备读取。非索引压缩层保留原完整历史路径。该实现仍需将小索引表同步回 CPU、仍保留原 full-prefix workspace 容量，不能宣称完全无同步或常量显存；额外索引历史与 gather 工作区明确计入中央 host/device/pinned 预留，显式构造参数也绑定运行配置。reset 在全部状态写完后记录事件，下一 CUDA 流必须依赖该事件。

## 本地结果与真机边界

本轮冻结后组合回归覆盖 160 个相关测试文件：**2548 passed / 11 skipped / 117 GPU deselected**，Windows / Python 3.11 / CPU Torch，1020.96 秒。源码、测试和工具在整个回归期间摘要一致；保留一条既有 Windows ACL 子进程 UTF-8 解码警告，断言通过。中途发现的专家 D2H 计数、KV scratch 准入和跨流 reset 排序问题已修复并包含在这次最终回归中，早先中止的快照不作为通过证据。GPU SHA、原生 FP4 图、同机多卡、四模式、长上下文和生产吞吐均待 5090 验收。签名绑定报告与身份，不构成不可信远端 GPU 的数学执行证明。按用户新安排提交默认关闭的代码；代码发布不代表真机验收或性能资格通过。
