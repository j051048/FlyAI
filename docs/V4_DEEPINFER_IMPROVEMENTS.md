# DeepInfer 思路的本地落地与验收

核对日期：2026-10-21（含10月20日改造及本轮批执行修正）。本轮保留开放节点注册、区域优先、实测路线和原有环协议。参考
DeepInfer 提交 `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` 的设计，没有移入 SM100 内核，
也没有把 V4.1 权重接入原 V4 native ABI。没有新的 GPU 性能成绩。

| 优先级 | 本轮代码 | 边界 |
| --- | --- | --- |
| P0 | 签名的 host/CUDA 原始时间线、批请求轨迹诊断和独立 GPU gate | 时钟域独立；CPU fixture 不产生 GPU 资格 |
| P1 | 2–4 请求设备驻留状态、请求轴 native batch、有界 socket 在途和排空 | `/2`新greedy cohort；算子逐请求固定归约形状，非融合GEMM；相同前沿分组 |
| P1 | 置信度条件接受概率、留出集 ECE/Brier、按实际周期成本选择前缀 | 默认 shadow；绑定 workload/模型/源码/运行时/租约；不改接受数学 |
| P1 | 上下文/逐层成本表、实测时间配额控制 prefill quantum | 规划是预测；执行预算缺少完全匹配的测量时退回1 |
| P2 | 有效前沿精确缓存、pinned 拷贝生命周期、独立生成/评分与 teacher-forced 概率比较 | 不启用近似 replay；数据包哈希不是执行证明 |
| P2 | 控制 token 与普通文字分离、UTF-8增量解码、reasoning/DSML诊断 | API仍为纯文本；未实现的工具调用/schema显式拒绝 |

详见[设备批执行](V4_DEVICE_BATCHING.md)、[置信预算](V4_CONFIDENCE_BUDGET.md)、
[缓存与质量](V4_CACHE_QUALITY.md)。原 scalar whole/MoE-in-graph 配方保持原合同，不能把
`V4_CUDA_GRAPH=whole` 用到新 batch 后端。本轮独立实现可选的batch partial图：B1–B4、
上下文粗分桶、压缩分支，位置与输入每步刷新；attention/dense pre/post捕获，MoE与原生
exact-width Indexer保持eager。见[批图验收](V4_BATCH_GRAPH_ACCEPTANCE.md)。这是部分捕获，
没有批量whole/MoE-in-graph或GPU性能资格。

## 时间线

冻结配方前设置 `V4_RUNTIME_METRICS=1 V4_PROFILE_RUNTIME=1 V4_PROFILE_GPU_EVERY=16`，
可选 `V4_PROFILE_TIMELINE=2048`（0默认关闭，上限4096个span/收据）。普通 job 屏障才
解析未完成事件，不在 token 热路径同步。聚合计数保留全部测量，时间线保留最近有界片段并
报告丢弃数。逐层 span 含层号；未采样 GPU 执行不能从 host span 补造。

```bash
python phase0/v4_profile_report.py --report report.json --protocol protocol.json \
  --out diagnosis.json --trace-out timeline.json
```

`timeline.json` 可用 Perfetto/Chrome trace viewer 打开。每个签名收据、主机线程、CUDA stream
分别展示自己的epoch，不能直接对齐成跨节点关键路径，也不能相减得到CPU占比或宣称ZOS。

## 上下文与异构成本

`shard-stage-trace/3` 可在原trace加入 `cost_surface`。签名offer admission验证有界样本，
执行资源仍须匹配精确校准与租约。形状示例（不要把示例数字当成测量）：

```json
{
  "schema": "shard-stage-cost-surface/1",
  "source_sha256": "<校准engine_source_sha256清单canonical JSON的SHA256>",
  "samples": [{
    "context_tokens": 4096, "frame_tokens": 1, "warmness": "warm",
    "frame_samples_ms": [40, 41, 42, 40, 41],
    "prefill_samples_ms": [100, 102, 101, 100, 103],
    "layer_weights": [1, 2, 3, 4]
  }]
}
```

profile必须同时携带相同 `source_sha256`；offer admission与snapshot重查同runtime_config_sha256
校准中的实际源码清单，缺少或不同的绑定会隔离该surface。签名只是来源绑定，不是硬件attestation。
每形状5–128次完整stage wall测量、最多128种形状；每个所拥有层一个正权重，可由冻结
profiler的CUDA层区间汇总。权重仅分配完整stage成本的比例，不把CUDA时长当成独立wall time。
子段或较小上下文复用会标为预测，丢弃另一形状的percentile；更大上下文/另一帧宽不外推。
共享资源没有匹配忙时测量时保守按完整stage串行；P95/TTFT/TPOT仍需真实独立报告。

`V4_NATIVE_PREFILL_QUANTUM` 只是1–8能力上限；协调器 `native_lanes.prefill_quantum`
不能超过各段上限。`v4_prefill_budget.TimeBudget` 只使用绑定cohort/source/runtime/合同/
max-context/max-jobs、TTL、context/batch/quantum完全匹配的实测P95选择预算。压缩边界
可能使短上下文更慢，因此不拿大形状当小形状的性能保证。每轮检查取消/期限、decode优先、
prompt结束即停止该请求的额外quantum。`export_native_prefill_samples` 可从保守q1运行
导出真实连续轮跨度：同job集合/nonce/owner epoch/batch、连续frontier和cohort序号，且无
中间插入或缺轮，才用末轮drain时间减首轮submit时间；至少5次才保留该形状。它包括真实CPU
调用间隙，`measurement_scope` 明确声明连续跨度，不是单次advance(q)实验；不会用q1均值乘q
冒充测量。部署后仍须单变量对照确认收益。

## 下一次真机顺序

1. 用本轮源码重新冻结计划；旧计划不能替新代码背书。先按[5090完整验收](V4_5090_ACCEPTANCE.md)
   跑原scalar whole的回滚、一致性、同机多卡、四策略、生命周期与soak，原始失败也归档。
2. `phase0/v4_release.py gpu-check` mandatory list包含真实CUDA 2/3/4请求状态与
   token/top-K/logprob轨迹、B1–B4 partial图freshness/状态，以及warmup失败恢复和图数量预算。
   跳过/缺测不算通过。tiny native gate不能替代原生FP4 checkpoint批一致性；按
   [批图验收命令](V4_BATCH_GRAPH_ACCEPTANCE.md)分别跑scalar、native eager、partial对照。
3. 新batch cohort独立记录batch/singleton次数、峰值在途、host状态拷贝字节、共享资源预算、
   每请求收据、取消排空、slot复用和多并发P50/P95。与scalar比较固定prompt、长度、cold/warm，
   不把请求吞吐乘数当成单请求decode tok/s。
4. 置信预算先采固定cap签名报告，再离线校准、shadow。身份/租约变更后重新生成；以实际
   committed output / 完整周期取胜，不能只报告g。缓存和时间预算分别做单变量对照。
5. 导出generation JSONL，独立比较token和真实teacher-forced logprob；复核长上下文缓存、
   ring-wrap、rollback和UTF-8边界。回收协议/输入/收据/trace/工具与源码SHA256，再决定发布。

本轮开发验收只运行本地CPU/socket/HTTP测试，没有运行或租用GPU，也未修改生产服务；
用户随后单独授权提交推送源码，此授权不替代真机验收。
可选批图已接入但硬件资格待验；V4.1适配、近似replay和跨WAN专家并行仍未因本轮改动获得支持。

2026-10-20新增单份权重批诊断的固定CPU BF16 fixture中，B=4出现top-5及部分KV容差差异，而greedy
token相同；诊断如实报告失败，没有放宽阈值。不能据B=2/3的fixture结果推定B=4或真实FP4
通过。新路径默认关闭，2/3/4必须分别验收；微小logit差异下argmax/top-K改变也会使CLI失败。

## 2026-10-21：B4修正和独立批图

原B4偏差来自随请求行数改变的算子归约/近似路径，微小FP32 HC投影差异及Gate softplus
差异会被后续BF16状态放大。本轮新增 `v4-native-request-row-math/1` 域，只对batch内部
投影、收缩、逐元素和归约保持逐请求原算形；原scalar oracle、权重量化字节与验收阈值保持
独立。一次stage batch仍处理`[B,1,…]`，内部部分算子分行执行，不能宣称融合grouped GEMM。
批合同升级为`v4-native-small-batch-greedy/2`，旧`/1`校准或计划不能直接复用。

原seed7/17位置B4冻结轨迹已在零容差下通过；另以0、7、23三个seed×B2/B3/B4×37位置
核对输出、logits/logprobs、ordered top-K及有效KV，CPU fixture严格一致。near-tie和非有限值
注入仍会失败。这修正了10月20日的失败，不删除它，也不把CPU结果提升为真实FP4/GPU资格。

批图以`V4_NATIVE_BATCH_GRAPH=partial`和显式MIB/LIMIT启用，保持`V4_CUDA_GRAPH=0`。
原生Indexer在eager中按真实宽度选top-K、保留原顺序，末尾补`-1`后作为固定图输入；图键为
B、粗上下文桶与压缩分支，不为每个真实读取宽度建图。服务纯prefill签名绑定
`capture_graph=false`；首次产生输出的transition及后续decode允许捕获，图数量/字节到顶显式回落，无LRU。捕获和预热
事务均恢复工作区；不能确认流排空或恢复时拒绝提交状态，不能继续eager重试。

alias恢复或canonical设备拷贝失败会标记`batch_fault`，失败事务不发布新frontier，后续步骤
一律拒绝；必须重启本轮owned Stage并重建认证绑定。`clear()`、新owner epoch或普通重试
不能清除故障，也不能把发生部分拷贝的存储当作仍然精确的旧状态。

诊断绑定不可变`batch_runtime.graph_policy`，实际`batch_graphs`独立报告partial范围、
Indexer/MoE eager、捕获、回放与回落。实际CUDA请求partial而未观察到捕获不能通过图诊断；
CPU参考仅可证明算术。对外速度、TTFT/P95及长上下文仍需租机实测，无新增tok/s声明。

## 本地验证记录

环境：Windows、工作区Python3.11.15 / Torch2.14.1+cpu；`PYTHONUTF8=1`、
`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。本地文件列表为
`.venv/flyai-deepinfer-regression-files.json`（最终133个相关文件；不是全仓CI）。

- 扩展的131文件初次组合：2190 passed、11 skipped、84 GPU deselected，6 failed。
  运行期间源码尚在修改，3项触发源码防变更、1项读取了旧模块清单；另外2项为收据超时/断链。
  没有放宽源码检查、签名、数学或超时；停止编辑后在全新进程复跑这6项，全部通过。
- 最终新增batch/严格TCP/单权重诊断/时间预算/时间线/签名offer成本专项：58 passed，
  5 GPU deselected。另一次集成专项62 passed、5 GPU deselected，覆盖release/资源/审计；
  这些范围重叠，不能相加成独立通过数量。
- Python语法和修改文档的本地链接检查、`git diff --check`通过。

原始JUnit保存在本机`.venv/flyai-deepinfer-{local,recheck,integration,final-new-paths}-20261020.junit.xml`。
保留首次失败证据；不把复测登记为一次全组合无失败。CUDA/FP4、真实tok/s、完整模型质量和
长时稳定性均未由本地结果资格化。这份测试记录生成时Git基线为`cd7642e`，尚未commit、push。

2026-10-21本轮批修正与图桶的阶段性核心回归为154 passed / 10 GPU deselected；独立
contract/native-batch/release范围为67 passed / 6 deselected，JUnit
`.venv/v4-batch-graph-contract-final-20261021.junit.xml`。这些是重叠且有明确源码时点的
CPU专项，不相加成全仓CI或最终集成总数；后续冻结源码组合回归另行登记。

2026-10-21最终冻结源码组合回归：**137个相关测试文件，2333 passed、11 skipped、
92 GPU deselected，0 failed / 0 errors**，耗时703.09秒。使用上述CPU环境与`-m "not gpu"`；
不是全仓CI，也没有执行真实GPU/FP4或租机跑分。此前的专项计数与此组合重叠，不相加。

本轮修复还收紧了诊断证据：`exact`按源dtype的原始字节比较（区分正负零），非法NaN/Inf
不能资格化，只有显式compressor score负无穷哨兵可匹配；显存配额同时核对allocator的
allocated/reserved峰值。B4长轨迹报告独立上限65536条，输入、状态、报告及工作区仍需显式
资源预检，容量上限不等于已通过完整长上下文验收。

最终证据保存在本机：

- `.venv/flyai-b4-graph-regression-20261021.junit.xml`：137文件组合原始JUnit。
- `.venv/flyai-b4-graph-regression-files-20261021.json`：实际测试文件列表。
- `.venv/flyai-b4-graph-local-evidence-20261021.json`：JUnit与测试文件哈希、Git基线及源码未变核对。
- `.venv/flyai-b4-graph-source-20261021.json`：执行源码清单，组合前后SHA256一致：
  `c05b52f76b48f0c50d90da346669c7b2d2f80c5929fe6ec72b055b1a42d7db37`。

这些本机证据文件位于忽略目录，不包含新的GPU通过声明。测试记录生成时Git HEAD为`cd7642e`，
尚未commit、push；用户随后已单独授权提交推送本轮源码。真机步骤见
[批图验收](V4_BATCH_GRAPH_ACCEPTANCE.md)，同时保留原whole回滚与四模式门槛。
