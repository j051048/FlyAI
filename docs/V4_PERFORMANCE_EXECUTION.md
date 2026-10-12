# V4 P0–P2 执行与服务改造

核对日期：2026-10-11。本文描述当前代码与操作入口，补充
[服务加固](V4_SERVICE_HARDENING.md)、[状态优化](V4_STATEFUL_OPTIMIZATIONS.md) 和
[固定基准](V4_BENCHMARK.md)。本轮按用户安排暂缓真实 GPU 验证；没有新增
RTX 5090 速度、WAN 性能或持续运行成绩。历史成绩仍按原日期和条件解释。

## 1. Stage 关键路径与启动内核选择（P0）

已有 `V4_PROFILE_RUNTIME=1` 与 `V4_PROFILE_GPU_EVERY` 继续记录实际 host 阶段和
抽样 CUDA stream 间隔。whole-layer 图现在分别观察输入准备、图 replay、图外 MoE
和尾部图；完成的 CUDA events 会复用。统计等待在收据快照边界进行，token 热路径
不为诊断调用全设备同步。`snapshot(wait=False)` 只读取已完成事件。

开启该观测后重新冻结协议、运行完整基准，再生成诊断：

```sh
python phase0/v4_profile_report.py --report report.json \
  --protocol protocol.json --out stage-diagnosis.json
```

诊断重新检查 stage 签名、nonce、层覆盖和冻结运行身份。host 数据包含提交与等待，
GPU 数据是抽样 stream 区间；嵌套/并行阶段不能相加，两个分母不能相减来推导 CPU
百分比。诊断本身不通过速度验收，也不将历史 632ms/28ms 外推成当前开销分解。

`V4_FP8_AUTOTUNE=eager|graph` 在 `V4_FP8_GEMV=auto` 时启用启动选择。必须选择与
实际执行方式一致的测量模式；默认 `0`。现有 FP8 内核只改变合法 tile/pipeline 形状，
保留 K 分块累加顺序。候选先经过当前进程的 `torch.equal` 原始内核检查，之后对
M=1/8/32 分别测量；只有所有代表形状都满足收益门槛才选择候选，否则保留 reference。
JIT/测量失败也保留 reference，图捕获期间不调优。

可选 `V4_KERNEL_TUNING_DIR` 保存以 GPU UUID/架构、驱动、Torch/CUDA/TileLang、
源码、输出 dtype 和测量模式区分的结果。捕获调用点单独进入 graph 测量范围；island
图外的 FP8 与 whole 模式的 eager MoE 不被标成 graph。缓存只复用计时，重启仍重新检查候选的数值结果。
内核时间不等于整环收益；验收入口是 `tests/test_v4_kernel_tuning.py` 的实际 CUDA
用例及固定基准。`V4_DSPARK_RESULT_BUFFER=1` 另可启用复用的小型 draft/confidence
结果缓冲，完整保留 int64 token，只做一次 packed D2H；默认关闭，GPU 收益待验。

## 2. 请求成本路由（P0）

[routing_cost.py](../shard/routing_cost.py) 与 [ring_pool.py](../shard/ring_pool.py)
将准备/验证/tokenize 放在预留之前，并对兼容的 READY 候选进行选择。同区域优先，
原会话环只在成本相同等条件下优先；拥堵原环不再直接排除其他可用候选。

完成并验过收据的请求自动提供本机服务观测：first-token、有效 decode 间隔、完成
开销，按上下文、执行模式和运行身份保留有限样本及 TTL。没有匹配新鲜证据时明确
退回 scalar 估计，不把缺失信息当作零成本；该预测不是端到端 p95 或 SLO 保证。
业务中的投机接受度变化仍可能影响预测，实际结果需持续观察。

缓存折扣须提供同租户/请求/cohort/runtime、完整精确前缀和真实恢复首 token 成本。
旧 `extended_shadow` 与新 native growing-prefix 都不会用理论比例编造折扣；后者
目前没有资格完整 prompt/logits 命中成本。未知或不适用时折扣为零。

## 3. 同一暖环上的有限投机预算（P1）

新 `round_speculation` 只影响尚未发送的候选前缀，保持加载的 B、W、F、验证形状和
目标专家计算。默认 shadow；enforce 需要已复核的有限档位和当前精确绑定，F 必须为1。
EMA/迟滞/TTL 以实测有效提交与耗时控制选择，缺失全环专家 DMA 成本保留 unknown。
不改变已经在途的帧、epoch 或接受规则；恢复 attempt 不沿用上一 attempt 的学习。

先在同一被选暖环中逐档使用原基准 `prepare --draft-prefix-cap` 冻结实验并 `run`，
保留各自独立 protocol/report。静态 cap 实验均关闭预测性 dprev，保留事实性 dnxt；
必须与其他同口径 cap 比较。正常 shadow 不改变原提示行为。

formation 可配置 `round_calibration_file: "private/round.json"`。该路径按网络配置
所在目录解析；服务自动导出 `round.json.binding.json`，记录真实选中环的绑定信息。
校准后本机原子发布，无需重启形成过程或人工抄写 fence：

```sh
python phase0/v4_round_calibration.py --reports r8.json r2.json \
  --protocols p8.json p2.json --binding private/round.json.binding.json \
  --publish-slot --out private/round.json
```

默认发布 shadow；明确 `--mode enforce` 才调整候选发送数量。服务在新请求边界
检查有界私有文件、原始签名、源码、模型、GPU 分配、TTL 和当前租约后更新。
错误文件保留原批准策略；活动请求不被替换。换环/新租约不能沿用旧绑定。
该路径只由本机操作者配置，聊天客户端不能上传策略。

## 4. 原生独立请求与真实增长前缀（P1）

新增模式 `v4-native-tokenwise-greedy/1` 是独立的模型 numerical cohort，明确从第一
个 prompt token 开始逐 token 调用原 `Stage.forward`。2–4 个请求按 token quantum
交错；每个请求有独立窗口 KV、压缩历史、indexer/compressor、前沿、取消和收据。
权重共享，GPU 目标 buffers 原地恢复，host 槽位与元数据/恢复工作区显式计入资源校准。

这是 **serial scalar lanes**，不是多个请求融合成一次 GEMM；复制状态可能使单请求
更慢。本轮实现了真实多请求交错、独立数学与服务接线，没有登记 aggregate speedup。
首版支持 greedy、显存常驻专家与 KV；拒绝 MTP、旧配方策略、KV tiering、sealed IDs
及历史相关的 reference bypass 组合。HTTP JSON/SSE 成功结果回显 numerical contract。

在已有有效配置中显式选择新 cohort，并合并：

```json
{
  "batching": {"enabled": true, "max_jobs": 2, "max_step_tokens": 1},
  "formations": [{
    "mode": "greedy",
    "native_lanes": {
      "enabled": true,
      "numerical_contract": "v4-native-tokenwise-greedy/1"
    }
  }]
}
```

这只是配置片段；formation 中仍须配置原模型目录、节点/路线、资源校准、租约和
签名暖机。`cohort.numeric_contract` 必须同值。每个节点设 `V4_NATIVE_LANES=2|3|4`、
正的 `V4_NATIVE_LANE_HOST_MIB`，可选 `V4_NATIVE_LANE_CACHE_TTL_S`。tail 不加载 MTP。
运行校准后重新发布绑定该配置的 offer/template，旧 calibration 不适用。

暖机实际交错执行多个请求、核对独立新收据和合同；收到异常容量声明会在分配作业前
拒绝。新 owner 重连通过专用的已签 open/reset 屏障更新所有 stage 的会话所有权，
不放宽通用 fence 校验；旧 owner 的前缀不会被新 owner 复用。

成功且验过收据后保留 consumed frontier；增长聊天在同租户、cohort、源码、固定
horizon、owner epoch 和精确 raw-token 前缀下恢复，再只执行新增 token，无 full-prefill
shadow。最后生成而尚未消费的 token 不冒充缓存状态。该数值合同的参考是独立
tokenwise 执行，不能宣称与旧 full-prompt prefill 的所有浮点结果位相同。旧缓存模式
继续保留 shadow；不符合前缀或 owner 的请求冷执行。

## 5. 持久化、恢复和实际复算（P2）

SQLite 的 `append_tokens` 与 `Job.commit_many` 支持1–64个已经结算的 token 在一个
事务中落盘，之后才公开；seq/cursor 仍按 token 递增。没有新增等待定时器。
串行 DSpark 的已结算批次接通该路径，EOS/max_new/64+bonus、原请求恢复和取消竞态
均保留原提交边界。pipelined 单 token 确认不为了批量提交而等待。
SQLite 删除空 WAL 的竞态按正常情况处理，不误判协调器所有权失效。

native 模式可选 `V4_NATIVE_LANE_STATE_DIR` 与正的 `V4_NATIVE_LANE_DISK_MIB`，使用
节点独立私有目录保存 JSON+safetensors 状态。没有 pickle。完整身份、布局、frontier、
内容哈希与 TTL 在恢复前核对；manifest 最后原子发布，Unix fsync、Windows私有 DACL
检查、临时生成和 old+new 峰值磁盘预算均已处理。先检查真实可用磁盘再写。

该磁盘预算是单 store 的空间上限和 free-space 检查，**不等于整个文件系统的常驻
共享磁盘租约**。其他进程仍可消耗磁盘；容量/free-space/ENOSPC 失败只放弃该次缓存并明确释放请求状态，
签名返回 `cache=None`，保留已验证完成作业，后续冷执行。权限、完整性或未知 I/O
异常仍拒绝，不能把它们当作正常 cache miss。
只有同 owner epoch/精确运行身份的节点本地 checkpoint 可以恢复；进程/整机故障
造成新的 lease/session epoch 时拒绝旧状态，仍可走 journal 原请求前缀重放。
本轮不声称跨节点 KV 迁移、分布式协调器 HA 或自动选主。

[v4_replay_audit.py](../engines/deepseek_v4/v4_replay_audit.py) 的
`NativeReplayAuditor` 在专用可信副本上真正调用原 Stage 数学。复算前验证原 stage
签名及完整输入/输出 leaf 序列与签名根的关系，冻结输入 buffers，检查模型载荷、源码、
配置、数值合同、horizon 和挑战 TTL；记录真实比较成本。生产副本必须加载完整哈希
验证的 stage artifacts，显式 CPU fixture 不能声明 GPU。

本机接口 `OpenNetworkService.run_native_replay_audit(..., auditor=..., chunks=...)`
按原质量策略 issue → 原生复算 → 验签 ingest；没有贡献者自报或公共 HTTP 捷径。
首版仅支持 raw BF16 greedy transcript，并重算 head embedding；拒绝 FP8/sealed wire，
范围是层前向，不核对 tail logits/argmax、MTP、采样或完整网络算术证明。
可信验证者、原输入/状态和独立算力仍由操作者提供；配置不会自动产生这些资源。

## 6. 计入准备成本的健康环重组（P2）

`form_one` 记录真实 perf_counter 准备/装载/图捕获/签名暖机总时间和实际部署计划。
顶层可选 `formation_evidence_dir` 保存形成结果。导出原始 strict 基准证据：

```sh
python phase0/v4_ring_economics.py --report strict-report.json \
  --formation formation.json --formation-result recorded-formation.json \
  --ttl-s 3600 --out economics.json
```

在已有 formation 配置上合并如下片段，替代候选各自也需已校准的 economics artifact：

```json
{
  "healthy_rebalance": {
    "enabled": true,
    "stability_window_s": 300,
    "cooldown_s": 600,
    "horizon_requests": 100,
    "minimum_gain_fraction": 0.15,
    "minimum_net_savings_s": 30
  },
  "economics": {"path": "economics.json", "sha256": "<exported-file-sha256>"}
}
```

默认关闭。仅使用本地预配置、固定 SHA 的原始报告，重新验签并核对模型、部署、配方
和负载。未来窗口的预测节省须覆盖真实准备总时间和已知缓存损失；有稳定窗口和冷却。
候选在发布前隔离准入，实际准备超预算、证据过期或身份变化就放弃发布；已绑定的旧
请求继续执行，不逐 token 换链。

当前经济证据严格限于冻结 prompt token hashes/mode/长度/输出预算的混合负载；
真实完成流量不匹配时 hold。缓存损失未知、adaptive/native lanes 或热装 round policy
均不能用固定报告触发重组。这里是可复核的固定负载预测，不是任意业务的性能保证。

## 7. 验证与发布

本轮111个相关测试文件的组合回归为 **1903 passed / 11 skipped / 69 GPU deselected**。
跳过包括7个 CUDA 条件用例、3个 POSIX 条件用例和1个退化 toy oracle；不登记为硬件证据。
最终 BF16 复算入口收紧及 CLI/runtime 同名导入冲突修复后，分别有46 passed / 1 skipped /
3 GPU deselected 的功能复核、88 passed / 1 skipped 的导入/校准/服务复核；两组重叠，
不与组合结果相加。运行时证据模块为 `v4_round_evidence.py`，CLI 名称保持不变，
两种搜索路径顺序都有独立子进程回归。完整 Windows 全仓检查包含
既有编码/POSIX/模拟计时失败，代表用例与独立 `070ccd6` 基线复现一致；上述组合不代替
Linux Python 3.12 全仓 CI。

CPU/socket/HTTP 回归覆盖实际原生数学、2-stage 严格签名连接、多个 HTTP 请求交错、
增长前缀、RAM 淘汰后的真实磁盘加载、owner 重连、SQLite 原子性及校准/经济证据篡改。
CPU 测试不替代 native FP4 CUDA 数值、内核速度或真实 WAN 验收。

native 收据包含实际 `job_complete` 运行声明与源码变更检查，但暂不附共享 stage 的
`runtime_metrics`，避免把交错其他请求的计数误标为单作业观测。stats 明示
`per_job_runtime_metrics=unavailable_shared_stage_counters`。旧模式的分相 profiler
仍按其原合同使用；新增 lane 的 per-job 性能观测需要独立资源和 GPU 资格。

`phase0/v4_release.py gpu-check` 已纳入启动 FP8 调优、draft 结果缓冲和独立 native
lane CUDA 状态门控。原始 FP4 gate 继续存在；Torch reference lane CUDA 用例只验证
状态隔离，不单独资格整个 native FP4 模型。跳过或 CPU 条件保持 unverified。
该命令只在操作者可独占的空闲设备上执行；本轮按用户安排未运行。

仍需在同一版本/权重/冻结工作负载下测单请求 committed decode、并发 goodput、
TTFT/TPOT p95/p99、内存与取消/故障恢复。四卡 >=40、六卡 >=30 的速度线没有改变。
单机八卡结果也不能登记为 WAN 成绩。
