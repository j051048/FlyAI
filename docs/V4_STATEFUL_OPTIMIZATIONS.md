# V4 请求边界与预填充状态优化

核对日期：2026-10-09，按当前工作区代码。三个功能都默认关闭，适用于现有原生
43 层 DeepSeek-V4-Flash 引擎。没有新的 GPU 吞吐结论，也不声明支持 V4.1 权重。

| 功能 | 生效位置 | 当前保证与限制 |
|---|---|---|
| 投机配方控制器 | 一个请求开始前 | 只选本地批准、与已选节点校准相容的有限配方；请求内不改配方 |
| 全环前缀状态缓存 | 完成预填充后的排空边界 | exact repeat 可恢复本机状态；扩展前缀每请求仍执行完整原始预填充参考 |
| 专家搬运流水 | RAM 专家模式的预填充 MoE | 真实路由确定后提前排队下一批本机 DMA；不切 token 或改变 Expert 数学 |

## 1. 入口与准入

服务配置使用 [v4_network_service.py](../engines/deepseek_v4/v4_network_service.py) 的
`shard-open-network/1` 文件。前两项配置放在已有 `formations[i]` 对象中；节点环境与
配额必须先写入实际校准和批准的启动模板。下面都是配置片段，不是完整部署文件。

```bash
python engines/deepseek_v4/v4_network_service.py --config network.json \
  --auth-file service-auth.json --host 127.0.0.1 --port 8000
```

控制面从已选 offer/calibration 和已承诺租约提供 `verified_runtime_configs`、
`verified_lease_fences`。HTTP 客户端不能自行提交这些证据或增加配方。
启用 policy 或 cache 时，每个 selected pipeline stage 都必须携带 64 位小写十六进制
`runtime_config_sha256`，并与控制面选出的精确校准配置实际 digest 匹配；缺失或不匹配即拒绝。
关闭这些功能时保留旧 plan 的兼容路径，不把客户端提供的摘要当作已校准能力。
strict 部署仍要求完整 model cohort、节点/协调器签名身份、真实 head/tail 路线和签名 warmup；
租约仍保护实际 GPU、共享主机 RAM/pinned 预算及进程生命周期。
具体部署见 [OPEN_INFERENCE_NETWORK](OPEN_INFERENCE_NETWORK.md) 和
[V4_OPERATIONS](V4_OPERATIONS.md)。旧静态 `v4_gateway.py` CLI 不提供这两项配置接线。

`runtime_config_payload` 绑定新的 cache reserves 与专家 FIFO 配置，源码身份也包含新 helper。
即使 checkpoint 字节相同，修改环境、配额或实现后也不能沿用旧运行配置摘要与资源校准；
应重新测量并更新已批准模板/offer，按正常租约准入启动新进程。

## 2. 请求边界投机控制器

[v4_request_features.py](../engines/deepseek_v4/v4_request_features.py) 校验有限配方与完整
已选节点配置；[speculation_policy.py](../shard/speculation_policy.py) 管理有界观察与选择。
例如，已有 formation 使用 `mode: greedy`，且真实 tail 已加载 MTP、每个 stage 的实际
rollback 容量都至少为 4 时，可批准：

```json
{
  "mode": "greedy",
  "speculation_policy": {
    "enabled": true,
    "baseline_id": "baseline",
    "recipes": [
      {"recipe_id": "baseline", "mode": "greedy"},
      {"recipe_id": "pipe4", "mode": "pipelined", "depth": 4, "floor": 1, "lazy": false}
    ],
    "learning": {
      "max_states": 128,
      "max_pending": 128,
      "sample_capacity": 16,
      "max_probe_requests": 16,
      "feedback_ttl_s": 600,
      "decision_ttl_s": 1800
    }
  }
}
```

`enabled` 缺省为 false。配方数量为 1–16，字段只允许 `recipe_id/mode/depth/floor/lazy`。
greedy 固定为 `depth=1, floor=1, lazy=false`；pipelined 的 depth 至少为 2，floor 不超过 depth，
并且不能超出任一节点的已校准 rollback 容量。baseline 必须与已有 backend 的实际
mode/depth/floor/lazy 相同。设已加载 draft block 为 `loaded_B`、批准的 pipeline depth 为 `W`，
还要求 `min(loaded_B, W - 1) <= 64`；服务核对真实 tail 回复的 draft 块不超过批准的
`loaded_B`，不能由 HTTP 宣称更大的能力。只支持 greedy 与 pipelined 菜单；不更改训练的 MTP block 宽度、
模型、量化、放置、正在飞行的帧或 lazy-hint 算法。能力 helper 在接近节点容量时支持已批准
的 greedy 回退，但当前 HTTP 自动策略请求在 prepare 层先预留 64 token 的投机余量；
没有这段余量的请求会先被拒绝，不能依赖服务自动切 greedy 来放宽容量。
客户端显式设置 `shard_mode: "greedy"` 可对该请求退出控制器，使用完整的已配置 greedy
上下文预算；这仍不能超过服务及所有 stage 的实际容量。

观察按认证租户、cohort、环/租约 generation、完整校准配置以及请求长度/工作负载桶隔离。
先测 baseline，再做有限探测；最小观察、保持期和增益门槛避免逐请求抖动。选择的配方在本请求
及其故障重放中固定，取消、失败、重放或不充分的计量不会训练策略。
签名结果有效但缺少可靠 sweep/计时字段时，可以保留该作业结果，策略记录
`invalid_measurement` 并丢弃学习样本，不伪造速度数据。
评分是实际 committed decode token 除以 decode 加排空时间；TTFT/预填充和收据 sweep 分开。
条件接受率只使用已判定预测，取消的未判定帧记成本。结果的
`optimizations.speculation_policy` 保留选择、反馈及评分定义，不是硬件或速度认证。

## 3. 全环前缀状态缓存

缓存必须同时在服务 formation 和每个 stage 启用。服务端片段：

```json
{
  "conversation_cache": {
    "enabled": true,
    "max_entries": 128,
    "ttl_s": 300,
    "extended_shadow": false
  }
}
```

每个 stage 在导入前显式配置，例如：

```bash
export V4_CONVERSATION_CACHE_MIB=1024
export V4_CONVERSATION_CACHE_GPU_MIB=256
export V4_CONVERSATION_CACHE_ENTRIES=4
export V4_CONVERSATION_CACHE_TTL_S=300
```

容量仅为示例，需替换为实测配额。host MIB 默认为 0，表示关闭；GPU MIB 默认为 0，
entries 默认为 4，TTL 默认为 300 秒。host quota 包含保留的 Tensor 和 Python 元数据预算；
GPU quota 为恢复 taps/rollback/MTP 动态元数据预留，并非在 GPU 中保留另一份完整 KV。
当前 stage 环境入口使用普通 host 快照，不额外开启 pinned 快照。
配额要与权重、现有 KV/专家池、prefill/shadow scratch 一起计入共享主机及 GPU 的准入，
不能把 host reserve 当作实际内存余量。`runtime_config_payload.conversation_cache` 必须与
实际环境相符；无完整节点校准及租约绑定时服务拒绝启用。

启用 cache 后，hash/字节对照按固定大小、最多 1 MiB 的块执行，避免把完整 GPU Tensor
一次性打包到 host。校验 workspace 在上述总 quota **内部**扣除：host 预留 `3 * chunk`，
非连续 GPU 数据的临时 pack 最多占一个 chunk，再加正常恢复元数据预算。
状态与资源观察的 `hash_workspace_host_reserved_bytes`、
`hash_workspace_gpu_max_reserved_bytes` 记录这些上限。快照只能使用扣除 workspace 后的
其余配额；放不下时缓存不命中/捕获回退，执行原始 prefill，而不是追加未预订的 RAM/VRAM。
关闭 cache 不追加这项校验 workspace。

[v4_conversation_cache.py](../engines/deepseek_v4/v4_conversation_cache.py) 保存全部有效 main
KV/Indexer 历史、Compressor 递归状态、taps、rollback 元数据，以及已加载 MTP 的 KV/前沿。
它不把只含窗口和递归状态的 rollback snapshot 当作完整前缀快照。
[v4_conversation_protocol.py](../engines/deepseek_v4/v4_conversation_protocol.py) 在排空且已提交的
prompt 边界执行全 stage prepare，再执行 commit。固定缓冲原位恢复；存储/权重改变、过期、
配额不足、节点缺项或部分恢复失败时清理事务并 reset 全环，重新执行原始完整预填充。
认证、签名或传输损坏作为错误处理，不能伪装成可接受的缓存命中。

缓存身份绑定认证租户、tokenizer/模板、精确 token 前缀、cohort/数值合同、源码与运行配置、
环 boot/owner fence 和各节点租约 fence；请求 horizon 与执行模式也参与匹配。
缓存只留在原节点，重启、换环、升级、换权重或新 fence 不复用旧状态。TTL、entry/byte quotas
和 LRU 限制保留规模。它是进程内缓存，不是持久会话存储、跨环 KV 迁移或协调器 HA。

exact repeat 只跳过同一匹配前缀的预填充，恢复缓存的首 token 结果后仍执行新的 decode。
单 token 作业不使用该快路径；缓存首 token 为 EOS 时仍做真实预填充。
普通聊天往往改变消息编码、模板和 token 前缀，因此不能据此承诺多轮对话加速。

`extended_shadow: true` 是显式实验：若找到较短的精确 parent，先恢复并执行候选 suffix，
再 reset 并执行本请求完整原始 prefill，逐 stage 比较状态和 tail reply 字节。
无论比较通过与否，当前请求保留完整参考状态；通过也不授权下一种 prompt/shape 直接增量运行。
本请求没有省掉完整 prefill，并增加候选与快照成本。它可为之后同一完整前缀的 exact repeat
留下参考快照，不是已认证的普通多轮增量引擎。

多环服务可在顶层另设 `session_affinity: true`；客户端可提供 `shard_session_id`，以
租户/cohort 为域优先路由到仍 READY 的原环。它只是有界路由偏好，不强行使用 DRAINING/FAILED
环，也不代替缓存身份校验。相同 session 名称不能穿透租户或版本隔离。
旧静态环池入口另有显式 `--session-affinity`，也只提供路由偏好，不自动启用状态缓存。

### 收据范围

缓存恢复不重用旧收据。每次作业仍使用新 nonce/job、当前 signer 与完整层覆盖核验；
新的签名收据可带 `conversation_restore` 声明，chunks 只计算本次真正执行的新 forward。
命中响应的 `proof.scope` 是 `fresh_suffix_with_signed_prefix_restore`。
这证明新的 suffix 收据及签名的前缀恢复声明已按合同校验，不证明该前缀刚刚重新执行、
缓存状态由诚实算术产生，或物理 GPU 已远程认证。不得把它记成完整新执行前缀的成本或证据。
边界见 [V4_TRUST_BOUNDARIES](V4_TRUST_BOUNDARIES.md)。

## 4. 有界专家 DMA 预填充流水

以下环境只用于 `V4_EXPERT_PLACEMENT=ram`，在导入前设置：

```bash
export V4_EXPERT_PLACEMENT=ram
export V4_PREFILL_EXPERT_PIPELINE=1
export V4_PREFILL_EXPERT_DEPTH=2
export V4_PREFILL_EXPERT_BATCH=0
```

开关默认为 0；depth 默认为 2、有效范围 1–4；batch 默认为 0，自动选
`max(1, cache.capacity // depth)`。有效 depth 不超过 `capacity // batch`，当前 batch 与
lookahead 的总槽位不超过该 pool 已分配的缓存。depth=1、单槽或显式 batch 没有足够
lookahead 空间时，明确回退原 demand DMA。非法参数拒绝；实际复制故障不静默回退。
必须另外配置并校准原有 expert-cache budget/slots 和 runtime reserve。

[v4_prefill_expert_pipeline.py](../engines/deepseek_v4/v4_prefill_expert_pipeline.py) 在本层真实
Gate 已完成后，为升序的确定专家列表维护有限 FIFO。它在当前专家计算前排队下一批原始
packed 权重/scale，复用 canonical pinned host banks、copy stream 和固定 slot。
没有新的权重池、pinned staging arena、CPU 补算、跨节点取专家或额外 GPU bank。
源权重重载必须经过等待 copy 完成的冷屏障；slot 消费者完成事件阻止过早驱逐。
取消/异常释放所有已预订租约。CUDA capture warmup 和 replay 不启动该流水。

该 batch 是专家搬运批次。每个 Expert 仍处理原本完整 rows，逻辑专家累加顺序及 hash duplicate
scatter 保持不变；token、attention、Compressor、HC 和原生 prefill 输入 shape 不重新切块。
已有历史预测预取是另一开关 `V4_EXPERT_PREFETCH`；此 FIFO 不预测或重新计算未来路由。
完整 GPU resident 模式不启用 FIFO，也不增加该功能的热路径调度。

`Stage.prefill_pipeline_config()` 仅包含稳定配置；`prefill_pipeline_status()` 的 per-pool
配置/观察记录实际 depth、槽位/字节预算、复制、取消及峰值。CPU emulation 明确标为
`cpu_reference`，DMA 字节为 0，不代表设备重叠。签名 runtime observations 保留这些观察；
配置摘要与动态命中/计数分开，不能用 requested=1 代替实际生效证据。

## 5. 后续 CUDA 与 Vast 验收

CPU 原始字节、完整 toy 模型状态及真实 socket/签名验证属于实现回归。可选 CUDA 用例需要
真实 DMA/计算和相应 TileLang/SM120 依赖；条件跳过的用例不算 GPU 验收。
例如，在独立、已授权且空闲的验收设备上执行：

```bash
python -m pytest tests/test_v4_prefill_expert_pipeline.py \
  tests/test_v4_conversation_cache_gpu.py -m gpu -q
```

还需对真实权重的缓存恢复、MTP/rollback、长上下文、取消/断链、租约 fence 和缓存配额压力
进行完整环验证。分开比较全关闭、单个功能开启及组合；冻结 checkpoint/source/config、GPU/driver、
分层和路线，分别记录 exact repeat 与冷/扩展 prompt 的 TTFT、有效 committed decode、排空、
sweep、实际专家 DMA/等待及 cache proof scope。shadow 的完整参考与额外 suffix 成本必须计入。

既有四张 5090 >=40、六张 >=30 committed decode tok/s 的门槛保持不变，仍需新的真实
集群证据。此文档不报告吞吐改善，见 [V4_BENCHMARK](V4_BENCHMARK.md) 的固定协议和
[V4_NEXT_PHASE](V4_NEXT_PHASE.md) 的硬件/持续服务验证范围。
