# GPT-OSS 部署、调度与测量

这轮改造覆盖 OpenClaw 2026-10-08 报告中的部署故障，以及实际使用的
`phase0/specpipe.py` 路径。V4 与 GPT-OSS 共用注册、资源租约、HTTP/SSE、
会话协议和部署清单；模型算子仍由各自引擎实现。

文档对齐于 2026-10-08 的 `c2ab623`。本轮 985 通过、3 跳过属于选定的
本地非 GPU 回归集合，不是全仓 CI 或新的集群性能收据。

## 下载与模型身份

GPU 节点使用完整仓库。手工部署工具需要 `deploy` 依赖；GPT-OSS 的
模型依赖属于 `gpt-oss` extra。CUDA torch/kernels 由节点按实际硬件安装，
运行时会在转换权重前拒绝反量化回退，并检查实际 packed 权重布局。

```sh
pip install -e '.[gpt-oss,deploy]'
python phase0/get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b --anonymous
# 已下载的平铺权重：只取固定版本元数据并核对文件，不重新下载。

python phase0/get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b --anonymous --verify-existing
python phase0/gpt_oss_manifest.py --model /root/models/gpt-oss-120b --out cohort.json
```

认证下载使用 `--token-file` 或 `--token-env`。非法凭据脱敏报错，不能悄悄
变成匿名。匿名模式明确禁用 HF 缓存凭据。下载一次解析不可变 commit；
临时文件经大小与摘要验证后原子落盘。`.shard-download.json` 绑定完整文件
内容，首次加载重新哈希实际文件，并拒绝残留根目录权重覆盖及未验证索引引用。
未引用的 `original/` 等备份可保留。

当前严格 GPT-OSS stage 和 coordinator 都要求本地目录具备
`.shard-download.json` 中列出的**整份下载快照**，首次加载校验每个列出的文件。
stage 只把本段模型层加载进 GPU，不代表它只下载本段权重到磁盘。
通用 `shard.fetch` 的分块拉取是另一条路径，尚不能用一个只含本段文件的目录
直接满足这套完整 inventory 合同；本轮没有实现 partial-inventory。

需要复用 Python API 时，先调用 `shard.download_inventory.verify_inventory(...,
verify_files=True)`，再将其返回的 `VerifiedInventory` 传给
`shard.gpt_oss_contract.build_cohort_from_inventory`。序列化 JSON、普通 dict、
`payload_integrity_verified: true` 声明或仅有 safetensors header 的 inventory
都不能作为此构造器的验证证据。构造结果是完整 `ModelCohort` 描述符；下载
清单绑定固定 repo/revision、文件大小与摘要，但本身不是发布者签名或硬件证明。
offer 与会话签名另绑定这些公开身份字段。

生成的 cohort 使用实际支持的合同：`gpt-oss-hf/1`、
`shard-pipeline-session/1`、`greedy-native-mxfp4/1` 和 `mxfp4`。
配置、权重身份或兼容字段不符，生产入口在加载前失败。

`validate_supported_cohort(cohort, config)` 检查实际 `model_type=gpt_oss`、
MXFP4 配置、禁止反量化、实际层数与四个支持字段。调用者另外比对原始
`config.json` SHA256 和完整下载清单。MXFP4 guard 在 Transformers 转换前
阻止自动反量化，转换后检查指定层的 packed 权重、scales、参数和 buffers。
`packed_layout_verified=True` 的范围是存储布局；`native_execution_verified`
仍为 false，不能据此宣称 GPU 内核输出或吞吐已验证。非 MXFP4 模型沿用
通用加载器，但不能使用上述 GPT-OSS 生产合同冒充支持。

## 统一部署清单

`shard-pipeline-plan/1` 同时供启动器、节点和协调器使用。必需字段：

* `ring_id`、完整 `model_cohort`、其 `cohort_id`、真实 `n_layers`、`nstages`。
* `stages`：有序的 `index/node_id/gpu_uuid/lo/hi/head/tail/endpoint/next_endpoint`，
  每个节点的 `signer_pubkey`。区间半开，必须完整覆盖模型且无重叠。
* `coordinator`：实际 `head/tail` 拨号地址及 `signer_pubkey`。

所有公钥来自已经使用的 Ed25519 节点或控制器身份。私钥仅留在本机。
`--split 20,8,8` 自动推导区间并检查总层数；已有 `--lo/--hi` 仍有边界验证。
清单与 CLI 切分冲突会在加载前失败。

严格 `specpipe.py` 默认要求 `--deployment-plan` 或节点本地租约 assignment；
直接启动旧环必须显式给 `--legacy-protocol`。严格生产计划要求完整 cohort，
并校验真实配置、完整权重、signer、端点及运行模板。当前 GPT-OSS 服务收据
验证器要求各 stage 的 receipt signer 不同；同机多 GPU 的资源合法性并不会
自动满足这个签名分配条件。

```sh
python -m shard.pipeline_plan --plan ring.json --config /root/models/gpt-oss-120b/config.json
python phase0/deploy_oss.py --plan ring.json --nodes deployment-nodes.json
```

`deployment-nodes.json` 的 `nodes` 按清单 node_id 索引，分别提供：
`ssh_target`（SSH alias 或 user@host）、可选 `ssh_port/ssh_key`、远端
`workspace/model/node_key`，可选 `coordinator_key/device/max_context/transport`。
`transport` 默认使用现有 libp2p sidecar。协调器在 head 的主机运行；其
`coordinator_key` 必须匹配清单中的控制器公钥。

可选 `tunnels` 各含 `ssh_target` 和 `forward`，例如
`127.0.0.1:30001:127.0.0.1:29501`。启动器以独立 argv 建立 `ssh -L`，开启
`ExitOnForwardFailure` 与 keepalive。环境数据经 SSH stdin 发送，不出现在命令参数。
节点按尾到头启动，最后通过真实推理和完整收据确认暖机。

手工进程管理使用 `.shard-processes` 的持久记录，只停止本次拥有的进程；
启动失败与清理未确认会保留 reservation/orphan，不能重复启动冒充空闲。
本地管理员可用 `python -m shard.managed_launch --name NAME recover` 恢复
记录，恢复会核对随机启动 token 与 process birth。正式网络使用节点资源租约，
手工 PID 管理不替代 GPU/RAM 预算租约。

原 `launch_oss/launch_ngram/launch_libp2p` 已移除清空所有 GPU/端口的行为，
环境走 stdin。它们明确使用旧协议兼容模式；正式会话用上述清单或控制服务。
这里不包含仍保留旧批量清理命令的 M2.5 `m25_scatter_pipe.py`；其隔离部署
边界见 [DEPLOY_M25.md](../phase0/DEPLOY_M25.md)。
旧单目录 V4 HTTP 包需要一起分发 `http_gateway.py` 与 `service_queue.py`。

## 开放网络服务

沿用 [开放推理网络合同](OPEN_INFERENCE_NETWORK.md) 的签名 offers、
node-leases、registry 和 `shard-open-network/1` 配置。

```sh
SHARD_TRANSPORT=libp2p SHARD_RECEIPTS=1 SHARD_COORDINATOR_KEY=/path/to/controller.key \
  python engines/gpt_oss/network_service.py --config network.json --auth-file tenants.json
```

`SHARD_COORDINATOR_KEY` 是本机受保护私钥文件的路径，必须匹配计划中的
协调器公钥；控制服务也可将已加载的同一控制器 key 对象注入 backend。
HTTP 默认仅监听 `127.0.0.1:8000`；非 loopback 地址必须同时提供
`--tls-cert` 和 `--tls-key`。`--auth-file` 必须提供 `keys`（API key→tenant）
和 `tenants`（每租户的 `max_active/requests_per_minute/tokens_per_minute`）映射；
每个 key 至少 16 字符，真实 key 只留在受保护文件。公开节点入网不意味着
HTTP 推理、节点资源控制或环邻居连接免认证。

每个 formation 提供完整 `cohort`、本地 `dir`、实际 `head/tail`、
`planning_calibration` 文件和 `measurements`。节点必须广告对应角色与层范围
的精确 `PlacementRequirements/runtime_config` 模板。调度器联合选择节点顺序
和已有模板，考虑真实 GPU 峰值、共享 RAM/pinned 预算及 stage/nstages 约束。
选择结果绑定运行配置 SHA；服务上下文上限取所有选中节点实际容量的最小值。

GPT-OSS 规划校准 schema 为 `gpt-oss-planning-calibration/1`，明确给出
checkpoint/config/node/GPU/runtime-config 身份、时间和 TTL、测量方法与证据、
`layer_vram_bytes/kv_bytes_per_layer/reserve_bytes/head_reserve_bytes/tail_reserve_bytes/load_peak_extra_bytes`
及 `cap_layers/layer_ms`。存储下界不能当运行峰值；缺真实测量不能填零冒充可用。
可通过 `gpt_oss_manifest.py --calibration FILE --profile-out FILE` 生成对应 profile。

多 token 验证使用 `shard-stage-trace/2`，绑定具体层范围、node/GPU/cohort、
运行配置、`frame_tokens=K+1`、`context_tokens` 和 `warmness`。formation 的
`K/depth` 必须与 `workload` 的 `draft_tokens/frame_tokens/depth` 一致，
并提供接受／作废帧的明确假设。所有规划数值仍是预测。

新路线 schema `shard-link-measurements/2` 记录逻辑 src/dst、实际
`route_id/channel/src_endpoint/dst_endpoint/dialer_id/dial_endpoint`、
可达性、测量时间、TTL、带宽和延迟语义。`rtt` 可明确选择保守计价或
`half_rtt_assumption`；真实单向测量使用 `one_way`，不能混算。
生产 formation 必须指定实际 `coordinator_id`，以及
`route_endpoints: {route_id: engine_connect_address}`；实际拨号地址必须匹配
测量。tail→coordinator 的逻辑返回边由 coordinator 拨入 tail 的返回 listener。
缺路线绑定会失败，不使用另一个端口的延迟。旧 v1 保留其保守 RTT 语义。

同公网 IP 可作为发现线索，合法有向链不要求 head 与所有节点组成双向星形。
任意贡献者可注册；区域优先、容量、可用模板和实测链路决定执行资格。
现有 sidecar/部署层配置实际连接，代码没有新增 DHT/NAT 栈。

## 会话与观测

双向 nonce challenge 签名绑定身份、计划、角色、索引、层区间及通道用途。
head 签发 owner grant，第二个协调器立即获得 BUSY；消息携带会话及 fencing
信息，旧帧不能换标签送给新会话。所有 first-frame 等待有上限。
tail 重连按 owner 配对，不能静默丢失 reset ack。严格 V4 使用真实 tail
endpoint，旧 `ret_relay` 仅用于明确兼容模式。

`CONFIG` 输出实际版本、source hash、量化配置、层范围、运行配置 SHA 和
margin 模式，不输出私钥、token 或 prompt。节点租约覆盖加载及空闲驻留过程。
长期空闲会话使用独立 head keepalive。

节点 telemetry 最多保留 256 个样本：CPU host 耗时、异步 CUDA-event GPU
耗时、发送队列等待，以及独立认证控制通道的 RTT p50/p95。计时不依赖跨机
时钟同步，也不为了收集事件同步 GPU。控制 RTT 包含控制处理开销，既不是
单向时间，也不是 `recv_wait`。`/metrics` 异步读取节点状态，不阻塞推理。
这些滚动观测不能自动冒充绑定 context/frame 的规划校准。

V4/GPT-OSS 共用 `shard/http_gateway.py`：认证租户可提交、查看、恢复和取消
自己的请求；提交可用 `Idempotency-Key`。SSE 恢复使用本 job 的
`Last-Event-ID: job_id:token_cursor:published_character_cursor`，同时绑定已经
发布的文本前缀。只有最终完整签名收据校验通过才标记成功；先前流出的 token
不是已完成证明。队列与恢复记录存于进程内，重启不会保留这些 job。

## 投机调优与验收

`--adaptive-depth` 保持 K 和验证形状，在排空边界根据近期接受率调整深度，
不超过初始窗口。`--adaptive-pipe` 是混合 K 实验：完整 greedy 对照与发布前
前缀检查确保当前请求一致，全部对照成本计入计时。默认关闭，不能把增加的
计算排除后宣称端到端提速。单形状校准的自动组环服务拒绝混合 K；固定 K
的自适应 depth 可选用。

`NGRAM_MARGIN=auto` 是默认策略；`fixed:64` 明确固定值，旧整数语义保留。

```sh
SHARD_TRANSPORT=libp2p SHARD_RECEIPTS=1 SHARD_COORDINATOR_KEY=/path/to/controller.key \
  python phase0/benchmark_oss.py --plan ring.json --model /root/models/gpt-oss-120b \
  --cases cases.json --repeats 5 --K 8 --depth 4 --ngram-n 2 --out measurements.json
python -m shard.performance measurements.json
```

cases 是数组，每项含 `name/workload/request`。workload 明确选择
`novel/copy/code/long_context`，request 使用 text chat messages 和 token 上限。
报告保存真实新增提交 token、首 token、decode、包含恢复与收据验证的服务总耗时、
原始完整最终收据、模型/配置/源码身份与 p50/p95。decode 排除恢复历史、预填充
首 token 和 EOS；按实际提交量计算 gain，完整接受 K 不再多算一个 token。
reasoning 模型的原始 completion token 计数包含其生成的 reasoning token。
报告分开统计复制与普通生成，不把 `recv_wait` 写成网络 RTT。

V4 原四卡 >=40 / 六卡 >=30 tok/s 及原始收据、token 对照验收继续使用
`v4_benchmark/v4_soak`。严格环运行时增加 `--deployment-plan` 和
`--coordinator-key`；硬件与 frozen protocol 必须匹配。

本轮 CPU 测试覆盖真实 socket/HTTP、签名、租约、完整小模型 V4 参考输出、
GPT-OSS 合成 oracle 和失败恢复。真实 MXFP4 GPU 执行、吞吐与持续稳定性需要
在集群上重测，不能用 CPU 数字通过硬件速度线。

2026-06-19 的 [GPT-OSS WAN 历史记录](receipts/gpt-oss-120b-wan-20260619.json)
报告约 40 tok/s，2026-08-02 的 [V4 历史矩阵](receipts/v4-flash-matrix-20260802.json)
汇总 30.15 tok/s。它们保留当时配置与观测范围，不能认证本轮严格协议、下载
校验、混合缓存或测量修正后的性能。
