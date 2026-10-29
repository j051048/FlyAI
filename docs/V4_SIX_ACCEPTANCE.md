# 六项性能改造的 5090 验收

更新：2026-10-29。本指南覆盖全环成本、同机直传、生产选环、持久批量状态、RAM 专家分段图和增量前缀/KV。CPU、本机合成 CUDA case、原生 checkpoint、完整 strict 环是不同级别的证据；前三者不能替代完整环的验收。本轮没有新增的真机速度或质量成绩。

## 1. 冻结源码和分配设备

使用独立目录中的完整源码验收包；先校验包，再准备当前源码计划。已有计划在任何代码、测试或工具变化后失效。准备计划只读取/哈希文件，不启动 GPU、不下载模型。

```bash
python tools/v4_six_acceptance.py prepare --out /data/flyai-evidence/six-plan.json
python tools/v4_six_acceptance.py local \
  --plan /data/flyai-evidence/six-plan.json \
  --out-dir /data/flyai-evidence/six-local-01
```

`local` 通过后仍报告 `native_gpu_verified=false`，六项原模型 A/B 均为 pending。命令、退出码、JUnit、日志和公开产物摘要写入独立证据目录；`private/` 是本次受管进程的控制数据，不应公开打包。

8×5090 的示例安排是 GPU 0–5 运行六段环、6 做数值诊断、7 做原有 GPU gate。新增专项顺序使用空闲 GPU 6 或 6、7，不能与该 GPU 正在运行的旧验收重叠。记录真实 UUID、PCIe 拓扑、驱动/CUDA/Torch/TileLang、模型 catalogue/tokenizer/source SHA，以及空闲 VRAM/RAM/锁页预算。同公网 IP 不能作为共享内存或同一主机的资格证明。

## 2. 先完成原有 whole 回滚、四模式和完整环

按 [5090 原有完整验收](V4_5090_ACCEPTANCE.md) 修改现有示例配置中的模型目录、层边界、Python 和 GPU 分配。不能直接使用示例边界推断自己的模型层数或可用容量。

```bash
python tools/v4_5090_acceptance.py plan \
  --config /data/flyai-config/5090.json \
  --out /data/flyai-evidence/baseline-plan.json
python tools/v4_5090_acceptance.py run \
  --plan /data/flyai-evidence/baseline-plan.json \
  --run-dir /data/flyai-evidence/baseline-01 \
  --confirm-gpus 0,1,2,3,4,5,6,7 --execute
python tools/v4_5090_acceptance.py verify \
  --plan /data/flyai-evidence/baseline-plan.json \
  --run-dir /data/flyai-evidence/baseline-01
```

保留 greedy、spec、dspark、pipelined 四模式，cold/warm 分开；保留 whole 对 eager twin 的完整状态/回滚对照、reset/cancel/reconnect/租约失效和至少一小时 soak。CPU kernel、跳过 CUDA case、图回退、仅 argmax 相等、合成小模型或 legacy 会话，都不能填作原生 strict 通过。4/6 卡历史性能门槛按原协议保留；本轮没有通过修改门槛获得成绩。

## 3. 新增 CUDA 专项

下面命令需要明确可独占的真实 CUDA 设备。组件脚本发现任何所选 GPU case 跳过即失败，并保留首次错误；它不会签发原模型或产品 release 资格。

```bash
python tools/v4_six_acceptance.py gpu-components \
  --plan /data/flyai-evidence/six-plan.json \
  --out-dir /data/flyai-evidence/six-cuda-01 \
  --confirm-gpus 6 --execute-gpu

CUDA_VISIBLE_DEVICES=6 python tools/v4_six_acceptance.py digest-probe \
  --device cuda:0 --execute-gpu --repeats 20 \
  --sizes 32768,8388608 --out-dir /data/flyai-evidence/digest-ab-01

CUDA_VISIBLE_DEVICES=6,7 python phase0/v4_host_transport_probe.py \
  --source-device cuda:0 --target-device cuda:1 --execute-gpu \
  --tokens 1 --dim 4096 --hc 4 --max-mib 16 --repeats 20 \
  --out /data/flyai-evidence/ipc-decode.json
CUDA_VISIBLE_DEVICES=6,7 python phase0/v4_host_transport_probe.py \
  --source-device cuda:0 --target-device cuda:1 --execute-gpu \
  --tokens 256 --dim 4096 --hc 4 --max-mib 64 --repeats 20 \
  --out /data/flyai-evidence/ipc-prefill.json
CUDA_VISIBLE_DEVICES=6,7 python phase0/v4_host_transport_probe.py \
  --source-device cuda:0 --target-device cuda:1 --execute-gpu \
  --digest-backend device_sha256 \
  --tokens 256 --dim 4096 --hc 4 --max-mib 64 --repeats 20 \
  --out /data/flyai-evidence/ipc-device-sha-prefill.json
```

设备 SHA 测试覆盖空输入、55/56/63/64 等填充分界、真实 GPU 数据变化、多 tensor/非连续输入与完整签名收据相等。`digest-probe` 交错比较真实 D2H+CPU SHA 和设备 SHA+32 字节回传，单独记录首次编译。它不覆盖完整 IPC 的控制、ACK 和收据处理，报告固定 `performance_qualified=false`。当前设备 SHA 是标准顺序 SHA-256，可能比 OpenSSL 慢，不能仅凭少了大 D2H 就推荐开启。

RAM 专家 CUDA case 是真实图捕获/回放和迁移生命周期，其中 dense stand-in 只验证固定槽位程序读取新输入、新专家；不是原生 FP4 数学或吞吐证明。持久 batch 和 indexed KV 的 tiny CUDA case 也只覆盖相应局部状态，原模型义务仍在下一节。KV 专项另覆盖 stream A reset → stream B forward，必须看到完整新 reset 事件依赖；CPU 测试不能代替这一项。设备摘要 IPC probe 在测量之外预热编译，但完整 GPU 目标字节校验、ACK 和控制开销仍计入每轮样本。

## 4. 六项逐个做原模型 A/B

所有比较都固定同一原生 checkpoint、prompt token、输出长度、采样策略、硬件、网络、上下文和数值合同。每臂重新校准实际角色/形状、冻结启动有效值，至少做 ABBA warm 重复；编译/冷启动单列。禁止同时改变多个开关后把全部收益归给一项。

| 项目 | A/B 与必须保留的原始证据 |
|---|---|
| 全环成本/投机 | 现有固定前缀策略与启用成本选择；完整 stage 身份/epoch/frame 覆盖、实际 DMA、草稿/回滚废功、首次 capture 和过期重探记录。缺段保持 unknown，不得填零；报告 committed tokens/wall time 和 P95 TPOT，保持接受规则和 token 一致。 |
| 同机 GPU 激活 | 相同 strict 环先 socket/CPU hash，再匹配真实主机/GPU 身份的私有 IPC 配置；device hash 臂采用 `digest_backend=device_sha256`，CPU 臂为 `cpu`。相邻真实字节根、ACK、回收、实际 backend/fallback、完整 ready-to-ready 时间和显存/RAM/SHM 分开记录。私有通道能力文件按节点提供，不能写进公开配方或把网络传来的 digest 声明当本地 actual bytes。 |
| 生产选环 | 精确 benchmark 仍用于结果验收；生产形状桶只用于候选预测与有界试运行。相同到达时间回放长短混合请求、跨区域 RTT/抖动/丢包变化和异构卡，记录预测误差、覆盖率、切换/加载摊销及满足延迟目标的成本；不能只测冻结 prompt 集合。 |
| 持久 batch/异长 | 相同请求分别单独执行和混合进度合批，保存每请求输出/完整 KV/Compressor/Indexer，测状态拷贝 bytes/token、平均有效 batch、取消后回收及 P95 TTFT/TPOT。scalar、原 equal-frontier、持久/ragged 和图路径分别标明，不能把 sequence 维伪作请求维。 |
| RAM 专家分段图 | canonical/exclusive 同容量各测原序执行、ready-first、分段图。混合命中、全命中、低命中、驱逐中取消、reload/过期 lease 和 whole 失败恢复均要跑；保存全部 tensor bytes，不只 token。固定 slot 里的专家必须多次变化，graph replay 读到新专家。`V4_EXPERT_READY_FIRST=1`、`V4_EXPERT_READY_MIB=8`、`V4_EXPERT_SEGMENT_GRAPH_MIB` 按实测预算；保持 `V4_MOE_IN_GRAPH=0` 和 resident KV。分段图预算是 Stage 总预算，按当前实现分配给主层；不能把 eager 调度报作完整 MoE graph。 |
| 增量前缀/KV | 先生成绑定源码/权重/tokenizer/数值合同的真实资格工件，再测追加多轮、分叉、压缩边界、回滚、取消和租户隔离。canary mismatch 必须关闭复用并完整 prefill，不可删除 shadow 直接“加速”。layer KV 以 `V4_KV_INDEXED_GATHER=0/1` 单变量比较，保留原 top-K 及其顺序；8K/32K/128K 仅在实际预算支持时运行，记录 TTFT、H2D/D2H、峰值资源和所有状态。 |

ready-first 为避免冷专家的保全等待挡住已就绪专家，会先提交热专家计算，再逐个接入冷专家，最后保持逻辑专家原顺序归并。暂存输出受 `V4_EXPERT_READY_MIB` 限制，大 prefill 超额时保留原流式折叠；它与预先批量占槽的专家 prefill producer 不同时工作，观察值须说明实际生效路径。

原模型 `/3` 的服务测试使用既有 [network service 部署](OPEN_INFERENCE_NETWORK.md) 与[批量准入](V4_PERFORMANCE_EXECUTION.md)，在实际批准的节点模板中设置以下值，预算按该节点重新测量，不能将示例数值当作已测能力：

```text
V4_NATIVE_BATCH=1
V4_NATIVE_BATCH_STORAGE=persistent
V4_NATIVE_LANES=2   # 分别以 2 和 4 重做一次
V4_NATIVE_BATCH_REFERENCE=0
V4_NATIVE_LANE_HOST_MIB=<实际正预算>
V4_NATIVE_LANE_DEVICE_MIB=<实际正预算>
V4_NATIVE_BATCH_GRAPH=off
V4_NATIVE_BATCH_GRAPH_MIB=0
V4_CUDA_GRAPH=0
```

formation 的 `native_lanes` 使用 `{"enabled":true,"numerical_contract":"v4-native-persistent-ragged-greedy/3"}`，model cohort 的数值合同也同值；顶层 `batching` 启用，`max_jobs` 与 lane 数一致、`max_step_tokens=1`，tail 不加载 MTP。发布新校准、签名暖机和租约后，用现有 `v4_network_service.py --config ... --auth-file ...` 启动。在同一实际权重服务上，先分别请求，再以不同到达时间并发重放相同长短 prompt，保留每请求输出、独立 nonce、原始收据及 `native_batch` observation。至少确认 `/3`、`state_full_copy_bytes=0`、`ragged_forward_calls>0`；仅成功返回文本不算命中新路径。它仍采用独立 request-row 数学，不能把一次 Stage.forward 声称为大批 Tensor Core GEMM。原 `/2` 的 `phase0/v4_batch_diagnostics.py` 命令不选择 `/3`，不能拿它的结果冒充持久异长批量验收。

严格报告使用现有 `phase0/v4_benchmark.py prepare/run/verify/compare`，参见[基准协议](V4_BENCHMARK.md)。每个 variant 都要新建启动、校准和协议，不能把旧收据拼给新配方。已有一对真实报告后，以下命令会重新检查两份收据再运行单变量比较，保留命令和退出码：

```bash
python tools/v4_six_acceptance.py strict-ab \
  --plan /data/flyai-evidence/six-plan.json \
  --before /data/flyai-evidence/ready-off-report.json \
  --after /data/flyai-evidence/ready-on-report.json \
  --before-protocol /data/flyai-evidence/ready-off-protocol.json \
  --after-protocol /data/flyai-evidence/ready-on-protocol.json \
  --vary env:V4_EXPERT_READY_FIRST \
  --out-dir /data/flyai-evidence/ready-ab-01
```

`--vary` 仍受原协议允许项限制，例如 `round:draft_prefix_cap`、单个 `env:V4_FLAG`、`network` 或 `source`；不是任意多项变化的豁免。退出码为零只说明比较程序完成，仍要核对增益、快路径实际计数、数值状态和副作用。

## 5. 收尾和推送条件

将六项 A/B 原始报告、协议、source/package SHA、硬件 UUID/拓扑、有效配方、冷/热耗时、完整状态比较、失败日志和原 whole/四模式/soak 结果一起归档。用原有 `tools/v4_5090_acceptance.py archive` 归档它拥有的原 campaign，新增专项按公开产物 manifest 收集，排除 private 能力、密钥和模型权重。签名收据绑定身份和报告字节，不是对远端计算正确性的密码学证明。

操作员逐项登记 passed/failed/pending、实际性能增益和适用条件；任何原模型义务缺失，保持 pending。本工具始终不推送 GitHub，也不会把局部通过自动升级为整个产品通过。按用户新安排可先发布默认关闭的代码；验收齐全后，维护者依据完整报告决定推荐配置和性能资格。


## Appendix: signed incremental-prefix qualification

Date: 2026-10-29. This guide covers the optional **resident scalar greedy**
qualification. The producer and loader reject `spec`, `dspark`, `pipelined`,
nonzero temperature, loaded MTP, service lanes and KV offload. The existing
four-mode ring acceptance remains required independently. No GPU result is
claimed by this guide or by the CPU reference tests.

### Evidence and identity

Use one frozen source bundle, checkpoint, model cohort, tokenizer/template,
tenant binding, owner epoch, lease fences, reset execution and stage assignment.
Changing any of these needs a new campaign. `max_pos` is the exact reset horizon;
it is not inferred from the current prefix. A qualification from one horizon
does not authorize another.

The campaign spec is JSON with exactly these fields:

```text
schema = "v4-incremental-prefix-campaign/1"
binding = the coordinator's actual ConversationIdentity before execution binding
source_sha256 = digest(v4_observability.source_inventory()) from the frozen bundle
execution = {max_pos, spec:false, dspark:false, pipelined:false, temp:0.0, seed}
max_context = largest tested prefix, no greater than the loaded horizon
max_suffix = largest tested append length
parent_alignment = the tested parent alignment
```

Obtain the binding and reset values from the actual coordinator. Do not invent
checkpoint/cohort/runtime identifiers or copy the CPU test fixture identifiers.

The pinned evidence JSON starts with:

```text
schema = "v4-incremental-prefix-evidence/1"
coordinator_pubkey = the public key pinned by the strict plan
stages = ordered [{stage, node_id, pubkey, lo, hi,
                  runtime_config_sha256, gpu_uuid}]
strict_canaries = []
native_cases = []
```

The stage ranges must tile the model. Every public key, GPU UUID and runtime
configuration digest comes from the actual strict deployment. The collector
verifies all signatures and assignment identities. Signatures identify who
reported measurements; they are not remote GPU computation proofs.

### Frozen per-stage inputs

Prepare a safetensors file from the same frozen target execution. It contains:

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `token_ids` | CPU int64 `[1,N]` | Teacher-forced target tokens. |
| `other_token_ids` | CPU int64 `[1,N]` | A distinct valid negative-control sequence. |
| `hidden` | Native Stage dtype `[1,N,hc_mult,dim]` | Required only for a non-head Stage, its frozen incoming hidden states. |
| `other_hidden` | Same type/shape | Required for the non-head negative-control sequence. |

Use inputs crossing an owned compression boundary. The producer records each
owned layer's compression ratio. A Stage with no compressor has no compressed
state branch to exercise. Neither the collector nor the producer converts a
failed or uncovered compression case into a pass.

### Run the owned native cases

The following command runs in its own process on an explicitly available GPU.
It loads one assigned Stage and reuses its weights sequentially. Repeat it for
every Stage with that Stage's actual range, inputs and GPU identity. Keep the
same engine environment as its strict worker; a runtime digest mismatch is a
refusal, not a reason to edit the pin.

```bash
CUDA_VISIBLE_DEVICES="$OWNED_GPU" python phase0/v4_incremental_case_probe.py \
  --spec "$CAMPAIGN/spec.json" --evidence "$CAMPAIGN/pins.json" \
  --checkpoint "$CHECKPOINT" --input "$INPUTS/stage-$STAGE.safetensors" \
  --stage-index "$STAGE" --lo "$LO" --hi "$HI" \
  --parent-count "$PARENT" --max-seq "$HORIZON" --device cuda:0 \
  --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB" \
  --key-file "$PRIVATE_COORDINATOR_KEY" --confirm-gpu \
  --out-dir "$EVIDENCE/native-stage-$STAGE"
```

The coordinator private key is read locally for signing and is never included
in output. Store it outside the source/evidence bundle with its normal private
permissions. Output directories must be new.

The producer executes full-prefill reference versus parent/suffix execution,
compression crossing, actual native `_seek` rollback, cancellation after actual
work followed by reset/restore, independent reset, and an unrelated sequence
between a parent snapshot and its continuation. It hashes actual readable KV,
Indexer and Compressor state and last outputs. A non-tail Stage compares its
last hidden output; a tail compares native head logits and greedy token.

`report.json` contains all comparisons and the first differing state key. Only
actual CUDA/TileLang runs with matching UUID, source, checkpoint and runtime
identity can emit signed `native_cases`. Only byte-equal cases are signed. A
mismatch returns exit code 2 and leaves the campaign pending; do not loosen the
tolerance or edit hashes. CPU `--fixture --device cpu` checks the producer flow
and emits **no signed GPU cases**, even when its arithmetic comparisons pass.

These are owned numerical lifecycle probes. They do not replace the whole-ring
transport cancellation, owner-disconnect and four-mode tests.

### Collect the strict append canary

Run a strict greedy parent request followed by its appended request with
`extended_shadow` enabled. Retain the original coordinator-signed
`shadow_begin` and `shadow_compare` controls, every original stage-signed compare
ACK, and the **same appended job's** complete signed receipts. The coordinator
records bounded controls/ACKs as `incremental_canary_evidence` when a shadow
comparison finishes. The receipts come from that request's final result.

Wrap the unmodified fields in a JSON file:

```text
{
  "schema": "v4-incremental-strict-canary/1",
  "canary": {
    "begin_request": original signed shadow_begin,
    "compare_request": original signed shadow_compare,
    "acks": original ordered signed comparison ACKs,
    "receipts": same-job complete signed receipts
  }
}
```

The collector checks the same job/nonce/transaction/prefix/execution, all-stage
comparison success, the BF16 activation chain, GPU UUIDs, TileLang/CUDA versions,
source inventory digest, runtime configuration digests and pinned signers.
Unsigned summaries, CPU observations, missing stages and another job's receipts
are rejected.

### Build the loadable policy

```bash
python phase0/v4_incremental_qualification.py \
  --spec "$CAMPAIGN/spec.json" --evidence "$CAMPAIGN/pins.json" \
  --canary-report "$EVIDENCE/strict-canary.json" \
  --case-report "$EVIDENCE/native-stage-0/report.json" \
  --case-report "$EVIDENCE/native-stage-1/report.json" \
  --out-dir "$EVIDENCE/qualification"
```

Repeat `--case-report` for all assigned Stages. The collector writes
`qualification.json` only when the strict GPU canary and all six native cases
per Stage verify. Otherwise it writes only `report.json`, lists missing cases
and exits 2. Do not pass the pending report to the service as a qualification.

Pass the resulting complete JSON object as the coordinator's
`conversation_cache.incremental_qualification`, with
`incremental_shadow_every` in `1..64`. The serving loader rechecks the embedded
evidence and its content digest. It rejects a hand-written `gpu_verified:true`
or `cases:true` object without complete signed evidence.

After loading, the first eligible request still performs a fresh full-prefill
canary. At most `incremental_shadow_every - 1` eligible reuses occur between
canaries. A failed canary disables reuse for that policy. Source, tenant,
execution, horizon and geometry mismatches keep the ordinary reference path;
an EOS suffix result also uses actual prefill so a one-token result keeps a
real nonzero-work receipt.

Record reused prefix tokens, newly computed tokens, restore time, full-prefill
time, TTFT and cancellation recovery. Numerical qualification does not establish
a speed gain. Qualify or recommend this path only after the matching native and
whole-ring hardware acceptance succeeds.
