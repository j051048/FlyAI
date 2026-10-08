# 节点权重准备与请求边界换环

本页描述本地参考实现的真实控制路径。CPU 测试覆盖实际 safetensors 文件、完整
SHA256、共享 SQLite 资源账本、签名 HTTP RPC 和本机子进程启动。它们不等于
真实 GPU 模型速度、数值精度或集群性能验收。

## 从计划到启动

控制面仍按同一模型 cohort、实际链路和已测量的阶段配置规划。准备权重不赋予
未校准的节点 GPU 容量或推理速度，也不跳过原有阶段租约。

启用准备的 formation 增加本机元数据路径：

```json
"weight_artifacts": {
  "catalog": "/srv/models/v4/.shard-model-artifacts.json",
  "pack": "/srv/models/v4/.shard-weight-pack.json"
}
```

这两个路径位于控制器本机；节点的源路径另行配置。控制器从完整逻辑目录和
源 packing 派生本段的层、边界参数与 MTP 依赖。网络 assignment 只带
`artifact_id`、`checkpoint_id`、`manifest_sha256` 三项固定身份，不带下载 URL、
目录、命令或环境变量。若规划器选定了 `preparation_mode`（`fetch` 或
`range_repack`），节点只执行本机已批准的该策略；现场资源不足会返回明确冲突以
重新规划，不会私自从重包改为更大的整容器获取。旧的无策略调用仍可使用受预算
约束的获取失败后重包路径。

新的逻辑目录 cohort 必须在公开 calibration 中附
`shard-storage-requirements/2`。它绑定本段源文件的真实路径、大小、SHA256、
artifact、角色和文件系统，并分别广告 fetch 与 range_repack 的可计算准备预算。
未来重包文件的 SHA256 未知，不能伪装成已验证；实际准备成功后才取得真实 witness。
节点 `resources.filesystems` 提供对应文件系统的测量容量与时间，不公开本机路径。
规划器按可行策略准入并合计共享磁盘、RAM 与 pinned RAM；不能仅因整个源容器
大于节点空闲盘就排除一个能够存下本段重包产物的节点。

[OpenNetworkService](../shard/network_service.py) 的加载流程先并行发送
`prepare_stage`，随后轮询 `prepare_status`。只有所有选中节点报告相符的已验证
READY 结果，才发送 `start_stage`。实际节点再次检查租约、校准模板和已准备目录；
远程调用者不能指定执行命令。

## 节点本机配置

[node-leases 控制服务](../shard/control_plane.py) 的 capacity JSON 使用已存在的
容量、cohort 和 `stages` 配置，并增加下面两项。所有路径属于该节点的文件系统。
本例占位符必须替换为真实 cohort、路径与内存域，不能直接复制成容量声明。

```json
{
  "filesystems": [
    {"filesystem_id": "model-disk", "path": "/srv/shard-cache"}
  ],
  "weight_sources": [
    {
      "cohort_id": "<ACTUAL_COHORT_SHA256>",
      "catalog": "/srv/source/.shard-model-artifacts.json",
      "pack": "/srv/source/.shard-weight-pack.json",
      "cache_root": "/srv/shard-cache/stages",
      "jobs_db": "/srv/shard-cache/preparation.sqlite",
      "provider": {"kind": "local", "root": "/srv/source"},
      "resources": {"filesystem_id": "model-disk"},
      "repack_resources": {"filesystem_id": "model-disk"},
      "repack": {"max_file_bytes": 536870912, "chunk_bytes": 1048576},
      "ttl_s": 120
    }
  ]
}
```

`provider` 也可为本机配置的 `{"kind":"http","base_url":"https://.../"}`，
使用已有 HTTP provider 的续传和跨源 Authorization 重定向保护。凭据只在受保护
的节点配置中；不能出现在公开 offer、目录或 assignment 中。

若转换结果已经分散到各节点，可配置 `kind=routed` 的逐文件源：

```json
{
  "kind": "routed",
  "routes": {
    "model000001-mp1.safetensors": {"kind": "local", "root": "/srv/node-a/v4"},
    "model000002-mp1.safetensors": {"kind": "http", "base_url": "https://<REAL_FILE_SERVER>/v4/"}
  },
  "default": {"kind": "local", "root": "/srv/model-assets"}
}
```

`default` 可服务配置、tokenizer 等资产。路径来自受钉定的 source pack，不从远程
assignment 接收 URI。范围读取和完整复制都使用同一逐文件路由。它不建立文件
服务器，也不会把 SSH alias 变成 HTTP 源；必须先有实际可用的本机挂载或文件
服务。这里复制的是准备阶段的权重文件，不是推理时跨节点请求专家。

本机 `stages[].argv` 必须用 `{model_dir}` 作为实际模型目录，例如
`["python3", "local-engine.py", "--dir", "{model_dir}", "--lo", "{lo}"]`。
它由 [LeasedProcessRunner](../shard/leased_runtime.py) 填入已验证目录，仍作为独立
argv 参数传递；不会交给 shell 解释。原有完整推理参数和校准环境仍须保持一致。

`resources` 与 `repack_resources` 可以提供非负字节数的显式
`disk_peak_bytes`、`ram_bytes`、`pinned_bytes` 上限。遗漏时，准备器从真实文件和
张量几何计算磁盘、元数据与有界 CPU I/O 缓冲的保守预算。这是准备工作的预算，
不能用来替代 GPU 推理校准。显式值低于预写入下界会在下载前拒绝。

## 准备状态、转换与发布

[WeightPrepareManager](../shard/weight_prepare.py) 将 job 与有限长度状态历史存入
SQLite：`planned → reserved → fetching/converting → verifying → published → ready`；
失败记录为 `failed`，已校验的临时下载可以复用。任务身份同时绑定节点、cohort、
源 artifact 与实际层/边界角色。

本机配置的 local/http/raw-repack worker 在 Python 进程内同步写入，并持有独占
OS job lock。重启恢复先取得同一缓存作用域、节点和 job 的锁，再确认该命名空间
的旧 worker 已结束并退休其持久 work handle；不会凭“进程重启”或远程布尔值
释放未知工作。未知的自定义/脱离子进程 converter 不能采用这个自动确认。
已经发布的成品先预留验证 RAM，再重新校验并幂等接入磁盘账本；不要求另有
第二份成品大小的空闲盘。

磁盘、RAM 和 pinned RAM 预留使用与已驻留 GPU 阶段相同的
[LeaseLedger](../shard/leases.py)。准备过期不能提前释放尚未结束的 worker；最终
缓存占盘与暂时资源分别记账。同一实际文件系统不能靠换标签获得另一份空间。
失败下载留下的 `.partial-*` 文件仍占实际磁盘空间。

普通获取只取本段所需的容器和资产。若整个源容器太大而资源预留失败，已配置的
`repack_resources` 会启用按张量范围读取与原始字节重包。HTTP 必须返回准确的
206 与 Content-Range；返回整个大文件不会被静默接受。每次范围读取有 chunk
上限，每个张量核验逻辑 SHA256，最终还核验文件、目录和阶段依赖。

重包保持逻辑 `checkpoint_id` 与目录根不变，物理 `artifact_id` 可以变化。
文件按层/MTP 分组；一个张量本身超过 `max_file_bytes` 时独占文件，因此该参数
不是绝对的文件尺寸上限。[repack_storage_bound](../shard/weight_artifacts.py)
将真实最大张量、头部、资产和三份元数据计入预算。

准备器只写自己的隐藏临时目录。完整校验之后在同一文件系统中原子改名发布，
保留经文件 inode/时间戳封印的真实哈希 witness，磁盘记账完成后才标为 READY。
已发布目录不会被覆写；后续发现损坏会拒绝加载，而不会修补正在使用的目录。
范围重包失败重试时可清理本 job 未发布的临时输出，不删除源模型或正在服务的模型。

## 资源不足与换环

`calibrated_alternatives` 是最多 16 个已有完整 formation 配置。每项有独立
`ring_id`，必须保持同一 cohort，仍经过实际规划、精确配置匹配、预留、准备和
签名预热。只有明确的资源冲突才会尝试下一项；哈希、身份、链路、校准等错误
不会被当成资源不足绕过。更小层段必须已经有相应校准配置，不能凭空缩造容量。

可用 `OpenNetworkService.replace(new_formation, old_ring_ids, aliases=...)`，
或在新的 formation 中设置 `replaces` 与 `publish_aliases`。
[RingPool.publish_replacement](../shard/ring_pool.py) 要求新环已经完成签名预热，
才在同一个请求准入边界发布新路由并将旧环置为 DRAINING。

已经绑定的排队、运行、幂等重试和 SSE 恢复请求保留原环、原 tokenizer 与原 cohort；
不会迁移正在执行的请求。旧环所有 bindings 和本地工作结束后，后台停止阶段，
只有实际清理确认后才释放旧租约。停止 RPC 慢或清理失败不阻塞新环准入，也不
把旧 GPU 宣称为可重用。已有 backend 的失败后原提示/前缀 replay 仍发生在原绑定
中，不跨 cohort 偷换执行。

换环需要可预留的新容量；旧环占满全部候选设备时，不能承诺无中断替换。
缓存内容、节点陈述和签名 RPC 都不构成远程硬件 attestation。

## 自动替换

包含 `calibrated_alternatives` 的 formation 默认启用 `auto_replace`；可显式设为
`false`。服务完成初始 formation 后启动独立维护线程，默认
`reconcile_interval_s=1`、`reconcile_cooldown_s=10`（可配置正有限秒数）。只对
FAILED、DRAINING、STOPPED 生命周期事件尝试替代，不用空闲显存读数把正在
驻留的健康环误判为资源不足。

先尝试已配置的候选 ID。临时容量失败后，默认 `auto_generations=true` 可在同一
批准模板上创建新的内部运行 epoch；只改变 `ring_id` 后缀，模型、校准、阶段
配置、源 artifacts、工作负载和路线均保持原样。失败尝试清理未确认时不继续
生成下一环；失败冷却、公开状态及有界运行记录避免不断占用内存。若只希望每个
候选 ID 尝试一次，可显式设置 `auto_generations=false`。

维护状态在 pool snapshot 的 `reconciliation` 中：准备、已发布、容量不可用、
清理未确认等会分开报告。不会为未测节点伪造容量，也不会停掉健康旧环来腾位。
签名、模型、缺少/过期校准和真实路径错误仍须解决；前置无计划只有在同角色、
同路线的反事实检查证明确实仅受已测资源不足阻止时，才会被归为可尝试备用
模板的容量错误。

## 初次流式转换

[v4_stream_prepare.py](../phase0/v4_stream_prepare.py) 可在转换源机器没有完整
167 GB 本地产物空间时，用有界 HF 输入缓存逐文件转换并交付各目标。示例：

```bash
python3 phase0/v4_stream_prepare.py convert \
  --repo deepseek-ai/DeepSeek-V4-Flash \
  --native-config vendor/deepseek_v4_ref/inference/config.json \
  --stage-targets stage-targets.json --scratch /srv/v4-scratch \
  --metadata-out /srv/v4-metadata --cohort-out v4-cohort.json \
  --cache-gib 16 --file-mib 512 --anonymous
```

`stage-targets.json` 是每项包含 `directory`、`lo`、`hi`、`head`、`tail`、`dspark`
的列表；显式 SSH 目标另外提供 `ssh_target` 与连接参数。所有目标必须是新目录，
层范围和尾节点 taps 按实际配置填写。此命令会向这些明确目标交付文件；本轮
本地测试没有执行网络下载或远程交付。

`--metadata-out` 只保留完整 GLOBAL/PACK、配置/tokenizer 等资产和 `outputs.json`；
不能声称它本机持有完整权重。`--cohort-out` 输出实际转换产生的 canonical cohort，
随后用它配置 formation 和节点来源。若今后改变切分，需要用 `outputs.json`
建立实际逐文件 provider 路由，不能假设每个目标仍有完整模型。

源缓存预算、一个待交付输出文件和 CPU 变换工作区仍必须能容纳；单一源文件超过
缓存上限会拒绝。单个输出张量超过 `--file-mib` 时独占文件，预算按真实张量计，
不能把 512 MiB 当作任何模型上都成立的绝对峰值。HF→native 的受支持变换保持
参考算术路径，native 范围重包只移动原始字节；CPU 文件测试不代替真实 GPU
模型数值验收。
