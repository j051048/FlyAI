# V4 服务加固与发布验收

核对日期：**2026-10-11**，按当前工作区实现。本文补充
[部署指南](V4_CLUSTER_DEPLOY_GUIDE.md)、[开放网络](OPEN_INFERENCE_NETWORK.md) 和
[状态优化](V4_STATEFUL_OPTIMIZATIONS.md)。这里只说明已接入的服务合同，不登记新的 GPU
速度或可靠性成绩。默认服务仍为单环串行、多环并行；新增规划、日志和调度策略由本机配置启用。

上一轮服务加固的选定 CPU/socket/HTTP 组合回归为 1813 passed / 8 skipped / 66 GPU deselected。
当前 P0–P2 的111文件相关组合为1903 passed / 11 skipped / 69 GPU deselected；最终
复算入口及独立 CLI/runtime 导入顺序另做专项复核，结果不累加，见 [执行改造](V4_PERFORMANCE_EXECUTION.md)。
用户提供的 CI 两项失败也已修复：选择性拉取测试改用真实 catalogue 与完整 payload hash，
不恢复无哈希的 legacy 通过路径；进程清理使用 Popen 唯一回收直接子进程，保留 SIGKILL
退出状态并继续等待真实后代。补充部署/资源回归为 218 passed / 1 POSIX skip；对应进程
套件在 WSL Ubuntu 22.04 / Python 3.10 为 51 passed。结果有范围与重叠，不相加；
完整 GitHub Python 3.12 CI 和实际 GPU 发布验收仍应在对应环境执行。

## 1. 配置位置与默认行为

完整仓库的严格入口为：

```sh
python engines/deepseek_v4/v4_network_service.py --config network.json \
  --auth-file service-auth.json --host 127.0.0.1 --port 8000
```

`network.json` 使用已有 `shard-open-network/1` 合同；模型 cohort、节点校准、租约、
真实路线和签名暖机仍是前提。公网 HTTP/控制面按部署指南配置 TLS。下面是**合并到已有
有效配置的片段**，没有省略部署前提的新捷径：

```json
{
  "job_journal": {
    "path": ".flyai-state/jobs.sqlite",
    "namespace": "v4-service",
    "max_jobs": 1024,
    "max_bytes": 67108864,
    "max_payload_bytes": 1048576,
    "max_tokens": 16384,
    "retention_s": 600,
    "busy_timeout_s": 5,
    "owner_ttl_s": 30
  },
  "scheduling": {"policy": "shortest_with_aging", "aging_s": 10},
  "batching": {"enabled": false},
  "session_affinity": false
}
```

这些字段在网络配置**顶层**，不能由贡献者 offer 或聊天请求开启。日志相对路径按
网络配置文件所在目录解析；默认建议目录 `.flyai-state/` 已在仓库忽略规则中，避免私有
prompt 数据被误提交。未配置日志时保留进程内队列；未配置 scheduling 时保留 FIFO
及租户轮转；未配置 batching 时关闭 token-step 调度。当前已接入的严格多环服务使用这些
选项，旧静态 gateway CLI 不提供相同的配置入口。

## 2. SLO、价格和共同故障域

`slo` 放入某个已有 `formations[i]`，只影响新组环的预测选择：

```json
{
  "slo": {
    "schema": "shard-planning-slo/1",
    "mode": "advisory",
    "targets": {"ttft_ms": 2000, "tpot_ms": 50, "observation_age_s": 30},
    "weights": {"ttft_ms": 1, "tpot_ms": 2, "hourly_cost": 0.2},
    "currency": "USD",
    "failure_domain_types": ["host"],
    "unknown_penalty": 10
  }
}
```

可用目标/权重还有 `request_cost`、`jitter_ms`、`packet_loss_rate`、
`failure_domain_concentration`；可显式提供同名 `normalizers`。
`advisory` 保留未知值及惩罚；`require_known` 拒绝**已设置目标或正权重**对应的未知指标和
超阈值预测，不要求所有无关指标齐全。不配置 slo 时维持原规划选择。

质量统计使用真实有向路线及其 src/dst、route ID、测量语义、时间与 TTL。原
`shard-link-measurements/1` 或 `/2` 的单条 edge 可附加：

```json
{
  "quality": {
    "schema": "shard-link-quality/1",
    "sample_count": 100,
    "window_s": 10,
    "latency_p95_ms": 20,
    "jitter_ms": 2,
    "packet_loss_rate": 0.001
  }
}
```

签名 offer 中已有 `models[i].profile.stage_trace` 可附加
`shard-stage-quality/1` 的 `sample_count/window_s/frame_p95_ms/prefill_p95_ms/queue_p95_ms`。
统计对象要求至少两个样本及正窗口，缺失统计量为 unknown；绑定必须匹配选中节点、GPU、
cohort、运行配置、层范围、帧形状、上下文和冷/暖状态。窗口起点计入观测年龄；普通无
新鲜度的 RTT 矩阵不能冒充尾延迟样本。TTFT 的传输预算还需要真实
`wire_bytes_per_token` 或与上下文匹配的 `prefill_context_tokens`，不能把固定旧
`prefill_bytes` 当作任意上下文的已知预算。

贡献者可以在**签名 offer 顶层**声明 `failure_domains`，例如
`{"host":"host-1","provider":"provider-a"}`，以及
`pricing={schema:"shard-node-price/1",currency:"USD",per_hour:1,measured_at:...,ttl_s:...}`。
价格需要同一货币且新鲜、覆盖所有选中节点；不做汇率换算。小时费用是选中计算资源的
报价之和；请求费用按预测占用时间估算，排除冷装载、空闲、网络、存储与实际结算条款。
签名只绑定声明者，不能认证真实物理域、硬件或账单。

共享故障域评分是相同域节点对所占比例，不是停机概率；同 IP 不会自动生成 host、
memory domain 或故障域。不同 GPU 同机仍允许参与。输出的 `planning.slo` 保留
`metrics/unknown_metrics/exceeded_targets/status`；无量纲 `selection_score` 与预测
`request_ms` 分开。组件 p95 放入有界资源日历是**压力场景**，不是端到端 p95 或 SLO
保证；丢包、抖动也不被转换成编造的 TCP 重传时间。搜寻仍有界，是否达标要实测。

## 3. 持久作业日志、重启与客户端断线

[job_journal.py](../shard/job_journal.py) 在本机 SQLite 记录原请求、精确 cohort、期限、
幂等键、逐 token 已提交前缀和终态结果。日志使用 owner lease 与递增 fence；失去所有权
的协调器拒绝继续提交或发布。一次 HTTP 写持有短事务，实际 write timeout 上限为
`min(请求配置值, owner_ttl_s/6, busy_timeout_s/2)`，避免慢客户端长期占锁；SQL_BUSY
心跳按有界预算重试，不将繁忙数据库当作已确认接管或无限等待。

恢复时从**原请求**重跑，并逐 token 核对已提交前缀；保持原 job ID、预算和截止时间。
恢复只能绑定原 cohort，不能用新 alias、不同 tokenizer 或另一模型顶替。后端还须支持
原请求前缀重放合同；V4 当前适配器有这一检查。若缺少兼容 READY 环，则作业保持
`recovering`，直到兼容环可用、期限到达或取消。不迁移 KV，也不继续未经核对的 speculative
状态。重放成本和次数受限，不是透明的零成本故障转移。

停止服务的开发 API 可用 `gateway.shutdown(preserve_pending=True)` 保留待恢复工作；
普通取消、期限和默认关闭流程按原语义终止作业。异常重启仍须确认旧 GPU/驻留进程的租约
与清理，不可仅因为 journal 接管成功就复用它们的资源。

HTTP 幂等键仍通过 `Idempotency-Key` 传入，作用域是认证租户。客户端可用同一原请求、同一
键重新 POST，并带最后收到的 SSE ID：

```text
Idempotency-Key: client-request-17
Last-Event-ID: <job-id>:<token-cursor>:<published-character-cursor>
```

服务从已提交前缀接续文本，保留字符游标以避免 UTF-8/tokenizer 边界重复；错误的 job、
token 或字符游标被拒绝。恢复重放与已发布前缀不符时失败，不把新文本接到旧输出。
成功完成后才返回最终 proof；流式 token 本身仍不是每步计算正确性的证明。客户端断线
触发原有“无其他客户端观察则取消”逻辑，**配置持久化不代表所有客户端断线都自动后台
继续**。需要保留的协调器停机与普通用户取消是不同操作。

Unix 路径要求当前用户私有目录 0700、文件 0600。Windows 为当前账户设置/核对私有 DACL，
拒绝允许其他账户访问的已有目录；因此最好让程序创建新的专用子目录，而非把日志放进
共享仓库根目录。路径不能有 symlink/reparse point。日志含私有原请求和 token，终态会
释放请求载荷、结果按 retention 保留；仍须按私有数据管理数据库与 WAL/备份。

SQLite 使用 FULL 同步、WAL、每 token 提交及有界结果预留；这会增加 CPU、磁盘 fsync 和
写锁开销。`max_jobs/max_bytes` 是容量限制，不能等同租户计费。单机 owner fencing 提供
排他恢复机制，**不是分布式共识、多副本协调器 HA、自动选主或支付结算**。恢复配置入口见
[network_service.py](../shard/network_service.py)，HTTP 细节见
[http_gateway.py](../shard/http_gateway.py)。

## 4. 当前可用的短请求调度与批处理边界

`shortest_with_aging` 已可用于当前 V4/GPT-OSS 多环队列：仍在租户轮转内选择作业，优先
`prompt_tokens + max_new` 较短的请求；等待超过 `aging_s` 后优先该租户最早等待的作业。
它调整下一请求顺序，不抢占正在运行的长请求，不切分当前 prefill，也不能保证固定
毫秒级等待上限。`session_affinity` 仅偏好同租户/同 cohort 的仍 READY 环，不绕过租约。

[continuous_batch.py](../shard/continuous_batch.py) 另提供默认关闭的 token-step 框架。
后端须实现 `batch_contract()/advance_batch()/release_batch_job()`，给出
`shard-token-batch/1` 的源码/配置、数值合同、独立每作业状态预算、
`max_prompt_tokens/max_context_tokens/max_new_tokens` 校准容量、prefill chunk、
per-job 取消及步长上限，并由签名暖机绑定合同摘要。最终结果仍逐作业验证新收据与提交
前缀；超步长、状态清理未确认或合同漂移会令环失败。

旧 V4/GPT-OSS 模式没有原生 qualified token-batch 合同，必须拒绝启用
`batching.enabled=true`。新增独立 `v4-native-tokenwise-greedy/1` cohort 可显式接入
2–4 请求标量时间片后端，包含真实 Stage 数学、HTTP、strict wire 和各自签名收据。
它不是多请求 GEMM，GPU 数值/吞吐资格仍待验。配置与增长前缀/持久化边界见
[执行改造](V4_PERFORMANCE_EXECUTION.md)；早期 CPU producer 测试本身仍不构成 GPU 证据。

## 5. 回放质量与缓存命中证明

顶层可配置 `quality_policy`，必须提供本机批准的 `validator_pubkeys`；其余有界选项见
[node_quality.py](../shard/node_quality.py)，例如：

```json
{
  "quality_policy": {
    "validator_pubkeys": ["<existing-validator-Ed25519-public-key-base64>"],
    "case_ttl_s": 60,
    "history_s": 3600,
    "failure_threshold": 2,
    "quarantine_s": 300
  }
}
```

本地控制代码通过 `issue_replay_audit` 固定原签名作业、节点/cohort、输入和执行身份，再由
批准的复算验证者比较实际输出字节并签署新 case。`ingest_replay_audit` 验签、校验 case
新鲜度/绑定和比较结论，去重并限制历史；达到失败阈值只筛除**新 formation**的该节点/cohort，
不阻止公开注册，也不随意取消其他 cohort 的既有请求。默认要求 `gpu_replay`；CPU 数学
测试不会被登记为 GPU 质量证据。报告是固定本地验证者的复算声明，不是贡献者自报质量分、
硬件 attestation 或完整模型执行的密码学证明。系统不会因配置了 policy 就自动获得可信
副本或完成所有作业复算；运营者仍须提供真实复算者和原输入/状态。

缓存命中现在跨 gateway→ring pool 保持
`fresh_suffix_with_signed_prefix_restore`，并重新验证正常签名、job/nonce/层覆盖/激活链及
每 stage 的 `conversation_restore` 与协调器恢复绑定；不能仅凭 `proof.verified=true`
放行，也不能把 fresh suffix 描述成重新计算过整个前缀。普通完整执行仍是
`complete_final_attempt`。同进程结果绑定防止结果在验证后被替换，不扩大收据的算术保证。
实现见 [execution_proof.py](../shard/execution_proof.py)、
[v4_gateway.py](../engines/deepseek_v4/v4_gateway.py) 和 [ring_pool.py](../shard/ring_pool.py)。

状态快照按当前有效 frontier 复制窗口、压缩历史、indexer 与 compressor 状态；不再把
整份未使用最大上下文容量当作有效 KV。固定 CUDA 目标/别名与图指针保持原合同，哈希按
逻辑张量字节有界分块，host/pinned 传输工作区计入配额。这降低无效复制/临时打包范围，
具体 GPU 收益仍待测；不能丢掉有状态压缩器或把复制完成声明当作诚实计算证明。

## 6. NativeTrace 的内部数值范围

[v4_native_trace.py](../engines/deepseek_v4/v4_native_trace.py) 的
`NativeTraceStore` 复用显式有界缓存，提供 `capture/restore/advance`。
`native-single-token-continuation/1` 只支持已排空 greedy、同一原始 prefill 加后续单 token
decode 的轨迹、原 tenant/cohort/精确消费 IDs 与固定 horizon。调用者持有 stage 执行锁；
非 head stage 必须提供原来的输入 hidden。MTP/speculation 或轨迹外状态变化被拒绝。

这不是公开 HTTP/wire 新聊天增量能力。decode 生成的状态在独立 trace 命名空间中，不可
冒充一次新的 full-prefill 快照；普通扩展聊天前缀仍走每请求完整原始预填充参考 shadow。
helper 的 `gpu_qualified=false` 保留未验收状态，CPU/条件 GPU 测试与生产 native FP4
发布 gate 分开。不得据此删掉 shadow 或宣称通用多轮 O(suffix) 加速。

## 7. 发布前的真实证据门槛

先按 [V4_BENCHMARK](V4_BENCHMARK.md) 固定模型/tokenizer/源码、真实四卡与六卡硬件、
运行配方、prompt IDs、route 和签名运行观察，并各自获取 `run` 原始报告。同请求 greedy
对照、TTFT/decode/排空/sweep 分开；不能用复制 prompt 高 g 或模拟结果替代正常负载。

[v4_release.py](../phase0/v4_release.py) 汇总四/六卡速度、持续运行和本机 GPU 门控。
先独立声明 soak 合同，示例 `soak-contract.json`：

```json
{
  "cycles": 25,
  "duration_s": 3600,
  "ttft_p95_s": 30,
  "token_gap_p95_s": 0.25,
  "max_idle_gap_s": 10
}
```

```sh
python phase0/v4_release.py freeze --four-protocol protocol-4.json \
  --six-protocol protocol-6.json --soak-protocol protocol-soak.json \
  --soak-contract soak-contract.json --out release-contract.json

# 仅在运营者自己的空闲、兼容 SM120 CUDA 设备上执行，需新的 JUnit 路径。
python phase0/v4_release.py gpu-check --junit gpu-gate.junit.xml --out gpu-gate.json

# 在已部署的严格环上运行；这里的门槛应与独立 soak 合同一致。
python -m phase0.v4_soak run --protocol protocol-soak.json --dir /models/v4 \
  --head 127.0.0.1:29610 --tail 127.0.0.1:29612 \
  --deployment-plan pipeline-plan.json --coordinator-key /private/coordinator.key \
  --cycles 25 --duration-s 3600 --ttft-p95-s 30 --token-gap-p95-s 0.25 \
  --max-idle-gap-s 10 --out soak.json

python phase0/v4_release.py verify --contract release-contract.json \
  --four report-4.json --six report-6.json --soak soak.json \
  --gpu-gate gpu-gate.json --out release-result.json
```

持续运行要求达到声明循环数与实际活动观察时长，每个候选请求有同 prompt 的 greedy
对照；空等时长、缺失样本、漂移身份、超 idle gap 或中断不能通过。GPU 门控要求原始
JUnit、稳定源码及所需 FP4/状态/图/continuation 用例；CPU 或跳过的 CUDA 用例返回
unverified，不能以测试 exit 0 凑通过。发布检查还比较 gate 与实际冻结节点的 GPU 类型、
驱动和依赖环境。

2026-10-11 多卡改造增加 whole 数值/回滚、island 实际捕获、原生量化 kernel 及 pinned
接收缓冲 CUDA 用例；GPU gate 在 owned 进程树监督下运行，超时不遗留自己的编译子进程。
命名配方与本地测试范围见 [V4_MULTIGPU_EXECUTION](V4_MULTIGPU_EXECUTION.md)。

固定目标仍为 **4×5090 ≥40、6×5090 ≥30 committed decode tok/s**，加独立持续运行
门槛；所有检查 passed 才接受。结果始终 `hardware_attested=false`：这是运营者实机证据，
不是远程硬件证明或分布式共识。本轮没有执行这些 GPU/远程命令，也没有新增实机达标
结论。历史论文、报告、收据保留原日期与原保证范围，不能由新增 verifier 反向认证。

## 8. 开发验证入口

可在 CPU 环境运行服务与规划专项：

```sh
python -m pytest -q tests/test_planning_slo.py tests/test_restored_ring_proof.py \
  tests/test_job_journal.py tests/test_durable_multi_service.py \
  tests/test_continuous_ring_batch.py tests/test_node_quality.py
python tools/check_references.py
```

这些测试分别验证预测、真实签名/SQLite/HTTP 控制流和独立状态 producer；不能合并为
原生 GPU 数值、速度或集群稳定性认证。本轮选定回归见本文开头，不替代全仓 CI；
不把之前构建的 1839/137 或重叠专项相加为本轮总数。
