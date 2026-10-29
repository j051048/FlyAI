# Strata-DualGPU 思路的六项本地执行改造

更新：2026-10-29。本轮针对已有 DeepSeek-V4-Flash native 引擎增加可选路径，默认关闭，保留原 scalar、whole 回滚、四种解码及独立 batch 合同。按用户新安排先发布默认关闭的代码，真机验收后再决定推荐配方；本轮没有新的 GPU 性能或质量成绩。

参考来源是 [Strata-DualGPU 固定源码快照](https://github.com/Hardin22/Strata-DualGPU/tree/03a5ec955ba152a5f499a18a0f2f2ac192309c28)。借鉴的是角色成本、内存层级、同步边界和草稿优化方法，未引入其 C++ 引擎、GGUF 内核、模型词表或速度数字。相同公网 IP 仍不能证明同一物理机，也不能授权 CUDA IPC 或共享 RAM。

## 已接入的六项内容

| 优先级 | 方向 | 当前实现与边界 |
|---|---|---|
| P0 | 按实际角色成本切层 | `shard-stage-cost-surface/2` 显式绑定 greedy/spec/dspark/pipelined 与 head/tail/draft。主层成本随所选层段缩放，Embedding、LM Head、MTP、DMA、sync 的已测固定成本保留。不预设强卡必须殿后。 |
| P0 | 可信本机激活传输 | strict 相邻 stage 可用私有配置选择本机 CUDA IPC/P2P；帧只携带绑定 job/nonce/epoch 的能力引用，不接收网络路径、pickle 或远端 IPC handle。接收方得到自己拥有的 tensor，完成后确认才可复用源槽位。 |
| P1 | GPU 热权重与 RAM 冷补集 | 新增 `exclusive` 专家存储；原 `canonical` 全 RAM 权重加 GPU 缓存保持默认。exclusive 在加载前建立热槽、冷补集和交换区，逐层计费，不先构造整份 CPU 权重再裁剪。 |
| P1 | 完整 committed-advance＋MTP 图 | 在已有 FAST seam 后增加有界完整草稿链：保持 M=1 中间提交推进，捕获最后一次完整 MTP 前向及 Head。位置、token、hidden 为活输入；捕获先备份，失败恢复或明确中止。 |
| P1 | 宿主 CPU／复制预算 | 同机进程通过同一个 SQLite budget 限制活跃 worker 与复制；调用线程可选 CPU affinity。复制许可持有至真正完成，未知完成不释放。不会推断 NUMA 页归属或更改进程全局 Torch 线程数。 |
| P2 | 原词表审计与离线草稿残差训练 | 使用原模型完整词表，支持真实 native MTP 特征采集、CPU 低秩残差蒸馏、按对话分组留出验证、工件哈希及模型/tokenizer 绑定。只改 draft logits，不改共享 target Head、验证者或接受规则。 |

2026-10-29 最终本地组合回归覆盖 148 个测试文件：**2446 passed / 11 skipped / 97 GPU deselected**，耗时 795.90 秒。测试前后的源码、测试和验收工具哈希一致；运行环境为 Windows、Python 3.11、CPU Torch。保留了 1 条 Windows ACL 子进程输出的 UTF-8 解码警告，相关测试断言通过。CPU 引用执行、真实进程／共享内存测试和优化器训练分别证明其本地合同，不证明 CUDA 捕获、原生 FP4 数值或速度，也不代替完整跨平台 CI。

## 成本测量和异构切层

`phase0/v4_role_calibration.py` 使用已验证工件和冻结输入，在自己拥有的 Stage 上运行实际前向，输出完成栅栏分割的 wall-time 成本面。它测量一次 teacher-forced full-commit frame，不测端到端请求吞吐或 speculative 接受率。至少五次有效样本，角色、上下文、frame 宽度和模式不能混装；运行时源码和原权重身份也必须匹配。

输入 safetensors 包含 `token_ids [1,context+frame_tokens]`；非 head 还需来自真实前驱的 `hidden [1,context+frame_tokens,hc_mult,dim]`，dtype 与 Stage 一致。草稿模式的尾段须持有全部 target layers 和实际 MTP。示例参数仅说明接口，层边界与预算须来自实际资源探针：

```bash
python phase0/v4_role_calibration.py \
  --checkpoint "$CHECKPOINT" --lo 36 --hi 43 --mode dspark \
  --input "$INPUTS/frozen-tail-k8.safetensors" --context 512 --frame-tokens 9 \
  --max-seq 8192 --repeats 5 --device cuda:0 \
  --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB" \
  --out "$EVIDENCE/tail-dspark-cost.json"
```

`phase0/v4_profile_report.py --role-cost-out` 也能从已验证签名报告导出带原始收据摘要的成本面；导出物仍需匹配运行时资源校准才可用于 offer。未拆分观察到的 DMA 仍包含在实际执行区间中；`dma=0` 不证明没有搬运，更不能把未解释时间全部称为 Python 开销。新成本预测也不是实际 SLO 证明。

## 私有本机传输

`V4_HOST_TRANSPORT_CONFIG` 指向操作员本机私有 JSON。Linux 文件和目录须属于运行用户且分别为 0600/0700；私钥、随机 secret、配置路径和 IPC handles 不进入 Git、公开计划或收据。配置绑定实际 cohort、相邻 signer 和 GPU UUID；不能根据公网 IP 自动生成授权。

配置字段如下，尖括号均为必须替换的说明值。`secret` 须在本机安全生成；示例不是可直接部署的配置：

```json
{
  "schema": "v4-host-transport-config/1",
  "host_id": "owned-rig-a",
  "directory": "/data/private-flyai/activation-slots",
  "secret": "<64个随机小写十六进制字符>",
  "backend": "cuda_ipc",
  "max_bytes": 16777216,
  "slots": 4,
  "ttl_s": 30,
  "channels": [{
    "id": "stage-0-to-1",
    "cohort_id": "<实际64位cohort摘要>",
    "src_pubkey": "<源stage的32字节Ed25519公钥base64>",
    "dst_pubkey": "<目标stage的32字节Ed25519公钥base64>",
    "src_uuid": "<源GPU UUID>",
    "dst_uuid": "<目标GPU UUID>"
  }]
}
```

`max_bytes` 是每 worker 源槽池总上限，单槽按 `max_bytes/slots` 分配；还要预算 CPU witness、hash/clone scratch 和元数据。CUDA driver 内存位于 Torch allocator 之外，资源库存单独计费。CUDA 不可用或 peer capability 不支持时可走有界共享内存；未匹配通道或池满时保留普通 socket 路径。必须看实际 IPC copy/fallback 计数，不能只看请求的 backend。

收据继续哈希实际激活字节。即使数据 D2D，哈希仍可能产生 D2H/CPU 工作，因此没有端到端零拷贝保证。未确认消费完成的槽位隔离保留，TTL 到期不能作为复用许可；进程或设备完成未知时保留现场和资源所有权。

## 专家冷补集与宿主预算

exclusive 首轮只用于 scalar、graph-off、驻留 KV。必须配置 `V4_EXPERT_PLACEMENT=ram`、`V4_EXPERT_STORAGE=exclusive`、正数 `V4_EXPERT_STORAGE_HOST_MIB` 和固定 `V4_EXPERT_CACHE_SLOTS`；`V4_EXPERT_CACHE_MIB=0`。服务加载要求原模型的已验证 checkpoint 身份。

若一个池有 N 个专家、H 个热槽、每专家 B 字节，热驻留为 `H*B`，冷 RAM 为 `(N-H)*B`，存在冷专家时另加 `B` 交换区。所有层和 MTP 池逐一累加，另计 resident、KV、工作区、锁页、加载峰值和余量。该 host quota 约束这些专家 bank，不是 Python、checkpoint mmap/page cache 或总 RSS 的硬限制。CPU emulation 中 hot 也是真实 CPU 内存，不能用它声称节省了主机 RAM。

迁移先等待旧读者，D2H 保全被逐出的唯一热副本，再执行 H2D、完成栅栏和所有权切换；加载、保存或取消也遵守 epoch/generation。它不是跨进程共享整份专家池，不提供跨节点专家副本；PP 不同层也不能假定可去重。显存节省、RAM 节省和 tok/s 需要分别验收，未命中时额外 D2H 可能降低速度。

宿主预算通过 `V4_HOST_COMPUTE_CONFIG` 单独开启。每个合作 worker 必须使用同一个绝对 `budget_path` 和相同容量；公开配置包含持久 `budget_id`，相同参数而不同数据库不能算同一预算。示例：

```json
{
  "schema": "v4-host-compute-config/1",
  "host_id": "owned-rig-a",
  "budget_path": "/data/private-flyai/host-budget.sqlite",
  "cpu_capacity": 16,
  "control_reserve": 2,
  "copy_capacity": 2,
  "workers": 2,
  "copies": 1,
  "cpu_ids": [],
  "strict_affinity": false,
  "ttl_s": 300,
  "wait_s": 0
}
```

在启动进程、import Torch 之前设置 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`，并核对 Torch intra-op 数不超过 workers。每次受控 CPU 调用占完整 workers 配额，不会在请求中调用 `torch.set_num_threads()`。affinity 只约束调用线程；没有 NUMA 页迁移或整个 Torch 线程池绑核保证。预算启用后复制可同步到完成再释放许可，可能损失原本 overlap，须单独 A/B。活跃许可不因 TTL 或异常自动回收，只有确认相关工作停止后才清理。

## MTP 图与草稿训练

完整草稿链需要 `V4_DSPARK_FAST=1`、`V4_DSPARK_CHAIN_GRAPH=1`、正数 `V4_DSPARK_CHAIN_MIB` 和有限 `V4_DSPARK_CHAIN_LIMIT`。原生 CUDA MTP 使用已知 pair MoE 后端时还需 `V4_DSPARK_MOE=1`；不支持的 FFN 会明确回退，不能按 graph 合格。生产 `V4_DSPARK_CHAIN_REFERENCE=0`，CPU 引用只能显式设 1。

图键按提交 run 长度与 window 读宽建立，受数量和字节上限约束，不做无限 LRU。重放读取新位置、新 token 和新 hidden；源码、权重、alias/rope 或 KV 存储变更后拒绝旧绑定。真实计数、分配量和失败状态放在 stage-lifetime observation，不能把它们作为单请求次数或写回已签名静态配置。

这条链只使用已经提交的 main taps。它没有实现 Strata 的未知未来窗口提前验证／回滚算法；已有 `pipelined` token 帧也不能等同于那个算法。先验证完整 MTP 图，再根据实际接受率和轮时评估 future-window 方向。

离线训练的完整 CLI 见[真机验收与训练步骤](V4_DUALGPU_ACCEPTANCE.md)。采集只使用每个 native 草稿块的 slot 0，避免把被拒绝后缀的未知父历史当 teacher 数据。训练用 CPU AdamW，对话 group 整组留出，原生词表完整保留；加载要求权重摘要、原模型/tokenizer/args 绑定和留出损失不退化，必须在任何草稿图捕获前安装一次。

导出工件的 `release_qualified=false`、`teacher_execution_proven=false`。本轮没有训练或发布真实 DeepSeek 适配器，没有 PPL、质量无损或接受率提升结论。训练损失下降不能替代完整 verifier/rollback、固定中文/代码/散文/工具负载和独立质量对照。

## 兼容性与默认值

| 组合 | 当前合同 |
|---|---|
| 不启用本轮新 flag | 保留原 scalar/whole、四种解码及资源合同；源码变化后仍须重新冻结校准和计划。 |
| strict scalar stage＋本机传输 | 已接入；仅私有配置的真实相邻节点，本机 UUID/身份验证仍必要。legacy 不接受本机优化授权。 |
| 原 canonical RAM 专家＋GPU cache | 原默认实现保留。 |
| exclusive＋scalar graph-off＋驻留 KV | 新 opt-in 路径；固定热槽与显式 host 预算；GPU pending。 |
| exclusive＋whole/island、layer KV 或 native batch | 原合同没有显式分段预算时拒绝。最新 opt-in 仅允许 whole-eager＋正的 `V4_EXPERT_SEGMENT_GRAPH_MIB`＋resident KV，见[六项验收](V4_SIX_ACCEPTANCE.md)；island、layer KV、native batch 组合仍拒绝。RAM 专家要求 `V4_MOE_IN_GRAPH=0 V4_DSPARK_MOE=0`。 |
| resident greedy MTP＋完整草稿链 | 新 opt-in 路径；已知可捕获 FFN 才能计作图覆盖；GPU/FP4 pending。 |
| 完整草稿链＋RAM 专家 | 拒绝。 |
| draft 残差＋原 target verifier | 安装在 draft Head；验证与接受规则保留。实际模型工件和四模式组合需要重新验收。 |
| 新路径＋独立 `/2` native batch | 不扩展其 MTP/host-expert 支持；原拒绝边界保留，参见[批图合同](V4_BATCH_GRAPH_ACCEPTANCE.md)。 |

新增环境变量为 `V4_HOST_TRANSPORT_CONFIG`、`V4_HOST_COMPUTE_CONFIG`、`V4_EXPERT_STORAGE`（默认 canonical）、`V4_EXPERT_STORAGE_HOST_MIB`、`V4_DSPARK_CHAIN_GRAPH`、`V4_DSPARK_CHAIN_MIB`、`V4_DSPARK_CHAIN_LIMIT`（默认 16）、`V4_DSPARK_CHAIN_REFERENCE`、`V4_DRAFT_ADAPTER_DIR`、`V4_DRAFT_ADAPTER_MIB`。未启用时路径为空、容量为零；不要仅保留容量而关闭其特性。全部变量必须在 import 前注入并纳入运行配置；不要带回已删除的 `V4_CHUNK_PREFILL` 或 `V4_PREFETCH`。

公开 benchmark `--env-file` 和一键验收的 environment 只填写公开开关；工具会冻结完整公开默认值并保持严格匹配。三个私有路径变量直接置于相应 worker 的私有启动环境，配置内容由已测 `runtime_config_sha256` 及内容摘要固定，不能放进公开配方。现有一键工具没有新增私有配置分发面；带私有 IPC、宿主预算或 adapter 的组合，先使用已启动的 strict 服务单独 A/B，不声称原一键命令会自动配置全部组合。

四卡 >=40、六卡 >=30 committed decode tok/s 仍是独立固定目标；容量提升不能代替速度线，synthetic GPU case、CPU fixtures 和签名字段也不能代替真实模型验收。操作入口见[本轮验收步骤](V4_DUALGPU_ACCEPTANCE.md)。
