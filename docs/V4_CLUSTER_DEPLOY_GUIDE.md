# FlyAI DeepSeek-V4 多机与多卡部署指南

基础严格部署合同来自 `c2ab623`（2026-10-08），本轮工作树新增有界权重准备与局部 artifact 加载，见第 2 节。模型仍是
`deepseek-ai/DeepSeek-V4-Flash-0731`：43 个目标层，DSpark 尾节点需要完整拥有
40、41、42 层以及 3 个 MTP 块。本指南不代表新集群已经通过 GPU 验收。
4×RTX 5090 ≥40、6×RTX 5090 ≥30 committed decode tok/s 仍待真实硬件验证。

共享 Hadamard 初始化、缓存别名、自检、strict SSH 手动部署及 P0–P2 复测步骤见
[V4_OPERATIONS](V4_OPERATIONS.md)。手动实验工具不代替下文的生产资源租约。

## 1. 先准备可审核的部署合同

当前生产路径使用 `shard-pipeline-plan/1`、完整 `model_cohort`、每个 stage 的
签名公钥，以及 `coordinator.signer_pubkey`。所有进程必须使用同一份执行合同；
角色、索引、层范围、cohort、用途和 expected peer 都经过双向挑战签名检查。
私钥不放进计划、报价、收据或日志。stage 的 `SHARD_NODE_KEY` 与其计划公钥对应；
协调器使用自己的 `--coordinator-key`，或 `SHARD_COORDINATOR_KEY` 文件路径。

开放网络的完整流程见 [OPEN_INFERENCE_NETWORK.md](OPEN_INFERENCE_NETWORK.md)：
签名报价 → 精确校准 → prepare/commit 租约 → 已批准的本机启动模板 → 签名 warmup → READY。
`LeasedProcessRunner` 持有常驻进程的资源，直到实际子进程退出后才确认清理。
只看到端口可连、GPU 空闲或一份遥测报告，不等于资源已经预订。

显存、RAM、可锁页内存、磁盘、加载峰值和共享 PCIe 必须按实际权重与运行配置计量。
不存在“32GB RAM 自动安全承载 4–6 层”或固定每层专家字节数的通用准入保证。
同主机的 RAM/pinned 预算合并核算；重复 GPU UUID 不能重复计容量。
磁盘需求由实际下载文件和临时转换空间决定，不能按层数假定只需 30–50GB。
详见 [RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md) 和 [COLOCATION_POLICY.md](COLOCATION_POLICY.md)。

## 2. 环境与权重

从完整 checkout 运行，使用 Python 3.11+：

```bash
git clone https://github.com/j051048/FlyAI.git
cd FlyAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[v4,deploy]"
```

项目 extras 安装共用依赖，不自动证明 CUDA PyTorch、TileLang 或驱动适合 RTX 5090。
在实际 sm120 主机上安装已验证的 CUDA/TileLang 组合，并保存精确版本和源码 hash。
旧 cu124 wheel、Python 3.10 或 CPU torch 都不是本指南的 RTX 5090 验收配置。
新 host 不应通过静默 BF16/CPU 回退冒充支持原始 FP4 内核。

V4 Stage 使用原生 reference 名称与 `ModelArgs` 配置。新路径可加载多个
`modelNNN-mp1.safetensors` 小文件及仅包含本 stage 参数的目录；只有 HF 下载目录仍不等于可直接加载。
当前有界流式转换、range 重打包、缓存/磁盘预算和发布步骤见
[WEIGHT_PREPARATION.md](WEIGHT_PREPARATION.md)。它保持 vendor 数学与 tensor 字节，
不要求 coordinator 保存完整巨型权重。各 stage 的文件与 tensor hash 必须通过后才可 READY。

原始转换器位于 `vendor/deepseek_v4_ref/inference/convert.py`，其参数是
`--hf-ckpt-path`、`--save-path`、`--n-experts`、`--model-parallel`、`--expert-dtype`；
该原始转换器保留全集 `state_dicts` 后才写出，不是有界流式入口；
`--model-parallel` 切的是 TP 专家/矩阵维度，不能拿它代替 pipeline 按层分片。
旧转换是单独的重操作，需要真实资源预算。参考版本见
[vendor provenance](../vendor/deepseek_v4_ref/PROVENANCE.md)。

冻结全局 tensor-root 与本地资产。完整 pack 实际校验全部 payload，stage 目录只校验
本段，coordinator 资产目录不声称完整权重存在；inventory 会分别记录这些验证范围：

```bash
python phase0/v4_benchmark.py inventory --dir /data/v4 --out local-identity.json
```

原生 artifact 使用全模型逻辑 tensor-root 作为 `checkpoint_id`，另用
`manifest_sha256` 绑定完整 catalog 与来源；两者不必相同。重打包保留两者和原参数名称。
局部目录拥有精确角色/层范围 manifest 与全部选中 tensor 的字节校验，不会把局部 headers
冒充完整模型身份。coordinator 可以只有 catalog、config、tokenizer，验证范围明确为资产，
不能报告“完整本地权重已校验”。发布目录不可原地覆盖；重复 tensor 名或已加载文件替换会拒绝。

单机多卡可以共享只读 checkpoint 文件，但每个 stage 的实际加载/专家/KV 池分别计预算。
部署规划器是 [shard/plan.py](../shard/plan.py)，它使用真实资源与测量合同。
选择性下载/重打包按真实 tensor 映射进行，不是把 167GB 平均除以 GPU 数；
tail 的 MTP、embedding/head、KV、graph 和临时量均需单独核算。
这一步准备和发布新环，不会改正在执行的环，也不代表 GPT-OSS 入口已支持 partial artifact。

## 3. 校准、KV 和专家缓存

RAM 专家池、缓存、KV 和查询分块是 opt-in。本段是配置示例，不是已过线配方：

```bash
export V4_DIR=/data/v4
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=2
export V4_EXPERT_CACHE_RESERVE_MIB=2048
export V4_MOE_IN_GRAPH=0
export V4_DSPARK_MOE=0
export V4_RUNTIME_METRICS=1
```

`V4_EXPERT_CACHE_SLOTS` 与总字节预算二选一。每层及 MTP 的路由专家保存在本机
pinned RAM，缓存未命中通过本机 DMA 进入 GPU；生产主路径没有 CPU 专家回退，
也没有跨节点取专家。更完整的限制见 [V4_HYBRID_RUNTIME.md](V4_HYBRID_RUNTIME.md)。
`V4_PREFILL_QUERY_CHUNK` 只分块注意力/Indexer 查询，保留投影、Compressor、HC、MoE
的原数学形状。`V4_KV_PLACEMENT=layer` 的预算与 graph 限制见 [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md)。

按实际 stage 的边界角色添加 `--head`、`--tail`、`--dspark`，再填入已确定的参数：

```bash
python engines/deepseek_v4/v4_resources.py measure --checkpoint /data/v4 \
  --lo 33 --hi 43 --tail --dspark --device cuda:0 --max-seq 8192 \
  --prefill-tokens 512 --decode-tokens 512 --output tail-observations.json
python -m shard.host_probe --devices 0,1,2,3 --host-id rig-a --dir /data/v4 \
  --pin-budget-mib 8192 --out rig-a-io.json
python -m shard.deployment measured-deployment.json --out deployment-check.json
```

这里的范围、长度、GPU ordinal 和 pin 数值仅展示合法参数，必须换成真实方案。
资源观察不自动构成完整校准；host probe 只验证本次分配/并发 I/O，不保留租约。
校准需要全部加载峰值、保留空间、来源、时间及 `runtime_config_payload`，其余必须明确未知。

## 4. 同机与跨机数据路线

同机 GPU 间可用 loopback/private endpoint；跨机优先已有 libp2p sidecar、可达内网或
经过实际测量的受控隧道。复用现有身份和 NAT 能力，不要求另装一套网络系统。
同公网 IP 不是同物理主机的证明，也不能直接推出 zero RTT 或 hairpin 一定不可用。
`-L`/`-R` 都可能形成合法路线；需要核对实际监听与拨号端，防止旧绑定把连接送错 stage。

生产默认允许同主机/子网的不同 GPU。host/subnet/adjacent_host 隔离是显式故障域或
WAN 实验策略；未知身份不能伪装成不同主机。新报价加入不改正在执行的环。
严格 V4 必须让 coordinator return 到达真正的 tail listener：当前 `--ret-relay`
旧 ingress 桥接只保留给显式 legacy 实验，不能默降级进 strict 环。

## 5. 当前生产 HTTP 入口

先完成本机模板、报价、测量、注册与 controller 配置，随后运行：

```bash
python engines/deepseek_v4/v4_network_service.py --config network.json \
  --auth-file /run/private/tenants.json --host 127.0.0.1 --port 8000
```

`network.json` 为 `shard-open-network/1`。该路径由共享 controller 启动批准的 leased
模板、生成 strict 计划、续租并要求真实签名 warmup。公开监听需要 TLS certificate/key。
Bearer 文件保护 HTTP；它不是 engine Ed25519 key。READY 后多个串行环可并发服务，
每个请求固定 backend/version；这不是 continuous batching 或持久化 HA。

`v4_gateway.py --deployment` 和 `--ring-pool` 是现有环的兼容入口。在 `c2ab623` 其
CLI loader 不注入 strict plan/controller key，故不应用来连接 strict stage。
不要把它的 deployment 资源检查混同于新的连接身份认证。详见 [V4_GATEWAY.md](V4_GATEWAY.md)。

## 6. 手工严格协议烟测

下面只是使用自己已分配硬件的手工启动示例；它不自动创建/续租节点资源。
假设校准后计划确实指定四个本机端口和示例范围，四份私钥已匹配计划：

```bash
CUDA_VISIBLE_DEVICES=3 SHARD_NODE_KEY=/run/private/stage3.key \
  python engines/deepseek_v4/v4_pipe.py stage --stage 3 --nstages 4 --lo 33 --hi 43 \
  --port 29623 --dir /data/v4 --dspark --receipts --deployment-plan pipeline-plan.json
CUDA_VISIBLE_DEVICES=2 SHARD_NODE_KEY=/run/private/stage2.key \
  python engines/deepseek_v4/v4_pipe.py stage --stage 2 --nstages 4 --lo 22 --hi 33 \
  --port 29622 --next 127.0.0.1:29623 --dir /data/v4 --receipts --deployment-plan pipeline-plan.json
CUDA_VISIBLE_DEVICES=1 SHARD_NODE_KEY=/run/private/stage1.key \
  python engines/deepseek_v4/v4_pipe.py stage --stage 1 --nstages 4 --lo 11 --hi 22 \
  --port 29621 --next 127.0.0.1:29622 --dir /data/v4 --receipts --deployment-plan pipeline-plan.json
CUDA_VISIBLE_DEVICES=0 SHARD_NODE_KEY=/run/private/stage0.key \
  python engines/deepseek_v4/v4_pipe.py stage --stage 0 --nstages 4 --lo 0 --hi 11 \
  --port 29610 --next 127.0.0.1:29621 --dir /data/v4 --receipts --deployment-plan pipeline-plan.json
```

各条命令在独立终端/监督器执行，或者只记录并回收自己启动的 PID；不能杀全部 GPU
进程或 `pkill -f` 清整个矿工环境。跨机时按同一计划填真实 `--next`、端口与绑定地址，
保留已认证/加密的数据路线，不能仅把监听暴露公网当作连接安全。

```bash
python engines/deepseek_v4/v4_pipe.py coord --dir /data/v4 --receipts \
  --deployment-plan pipeline-plan.json --coordinator-key /run/private/controller-receipt.key
```

该低层 CLI 使用 stdin NDJSON，参数名是 `jobId`、`maxNew`，不是 `max_new_tokens`。
请求示意（nonce 替换为新随机 32 字节的 64 位十六进制，swarmId 与实际计划一致）：

```json
{"jobId":"smoke-1","swarmId":"actual-ring-id","nonce":"<fresh-64-hex>","messages":[{"role":"user","content":"Briefly explain quantum computing."}],"maxNew":128,"dspark":true,"pipelined":true}
```

它输出 `SHARD_JOB_*` 控制记录；token 事件是提交进度计数。面向客户的文本/SSE、
认证、取消、幂等与完整 attempt 证据使用 HTTP 服务；低层 CLI 烟测不是生产验收报告。

## 7. 验收与排错

`python -m phase0.v4_acceptance` 只运行 CPU 分析/缓存模拟，不能证明硬件过线；
完整硬件速度验收使用下述冻结 benchmark 协议。
实际流程是 [V4_BENCHMARK.md](V4_BENCHMARK.md) 的冻结 suite + raw receipt/parity 检查，
再按 [V4_NEXT_PHASE.md](V4_NEXT_PHASE.md) 做独立 soak 合同。不同 warm/cold、提示词、
cohort 或环境的数字不能拼成达标报告。

握手错误先查完整 plan、双方 key、公钥和角色/端口；BUSY 表示已有远端 owner，不应
强占或杀别人的任务。Connection refused 查实际下一跳是否 READY 及有界 dial 重试。
OOM 查真实 resident/KV/cache/graph/load 峰值，而不是只改 `V4_EXPERT_PLACEMENT`。
pin 失败查整个主机可锁页预算与实际分配；不能仅凭一次小拷贝或盲目提高 `ulimit` 保证容量。
本轮 985 passed、3 skipped 是选定 CPU/socket 回归，不是全仓 CI 或新 GPU 集群成绩。
