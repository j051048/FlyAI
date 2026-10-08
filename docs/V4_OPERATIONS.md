# V4 运行自检、strict 部署与性能复测

2026-10-08，基于 `a6e96e3` 后的本地改造。适用原生 43 层 V4-Flash，
不是 V4.1 的兼容声明，也不代表新的 5090 集群成绩。

本轮统一的相关非 GPU 回归为 **1620 passed、3 skipped、41 GPU-marked deselected**，
耗时 435.10 秒；不是全仓 CI。两个完整 CPU 自检与真实本地 socket/HTTP/签名回归均在
这组结果内。GPU Hadamard、图重捕获/state 对照和实际集群 SLO 仍待执行。

## 1. 统一运行时初始化

服务、Stage 和 whole-layer benchmark 共用
[v4_runtime_init.py](../engines/deepseek_v4/v4_runtime_init.py)。
`V4_HADAMARD=auto` 优先实际扩展；扩展无法导入时使用 Torch 实现。
`extension` 强制扩展，`torch` 强制 Torch。实际选定函数及依赖摘要进入运行配置身份；
实际设备、BF16 和 Indexer 输入布局先执行确定性的数值 smoke。选择后的扩展若执行失败，
启动拒绝，不在活跃 CUDA graph 中悄悄切换后端。

这些 smoke 不等于所有真实输入的位匹配。Torch 在 CUDA 上执行仍需要单独测量；
CPU 数值、自逆误差或可导入状态都不能替代真实模型的 GPU 验收。

Attention、Indexer 和 Compressor 的 RoPE/cache 别名显式绑定并校验 storage、offset、
shape。原位 reset 保留视图；存储更换后重绑并使原捕获图失效，再捕获新图。
不能只把 `None` 替换掉而继续播放持有旧指针的图。

离线 CPU 全环检查使用独立进程：

```bash
python engines/deepseek_v4/v4_pipe.py selftest
python engines/deepseek_v4/v4_pipe.py selftest-relay
```

命令明确选择 CPU reference backend。已有 GPU kernel 的进程不能被该检查改写；
应重新启动自检进程。该检查不认证生产 GPU 内核、真实 checkpoint 或速度。

## 2. 手动 strict 实验部署

[v4_deploy.py](../phase0/v4_deploy.py) 复用现有 SSH 和 owned-process launcher。
生产租约服务继续使用 [开放网络入口](OPEN_INFERENCE_NETWORK.md)；本工具的 scope 是
`manual_strict_experiment_without_resource_leases`，不代表已预订 GPU/RAM/disk。

控制机只持完整 cohort、全局 catalog 与 coordinator 资产，各 stage 使用已经发布且不可原地
改写的本段目录，见 [WEIGHT_PREPARATION](WEIGHT_PREPARATION.md)。远端是完整 checkout，
使用同一个已安装环境的 Python。私钥留在各节点的受保护文件；配置引用路径，不包含密钥字节。

配置包含：

- `nodes`：SSH alias/target、可选 SSH port/key **路径**、workspace、python、model、
  node_key、实际 listen_port、device、max_context、可选环境及 expected_versions。
- `coordinator`：独立 node_id、signer_pubkey，以及在协调器视角可拨号的 head/tail。
  对应 node 可以另有 coordinator_key 路径。
- `stages`：按层顺序的 node_id、实际 gpu_uuid、signer_pubkey、endpoint、next_endpoint。
  `next_endpoint` 是发送节点视角；tail 的 next_endpoint 为 null。
- `split`：明确连续分层，不代替显存、主机内存或磁盘测量。
- `environment`：统一注入的显式字符串配置，覆盖冻结默认实验配方。
- `tunnel_routes`：可选的 caller-local SSH 路线。

身份、资源和端点要来自实际配置。3 张大卡或 6 张消费卡是否可用，必须由实际角色、
加载峰值和校准决定；不能把本模板当成均分容量保证。

```bash
python phase0/v4_deploy.py generate --metadata /data/v4-coordinator \
  --cohort cohort.json --config v4-deploy.json --ring-id v4-test-20261008 \
  --out plan.json
python phase0/v4_deploy.py preflight --plan plan.json --config v4-deploy.json
python phase0/v4_deploy.py start --plan plan.json --config v4-deploy.json \
  --startup-timeout 900
python phase0/v4_deploy.py health --plan plan.json --config v4-deploy.json
python phase0/v4_deploy.py stop --plan plan.json --config v4-deploy.json
```

预检汇总 config/cohort、真实 payload/阶段角色依赖、签名 key、tokenizer、版本、GPU UUID、
Hadamard 输入布局、监听冲突及主机资源。预览不会验证全部 payload/GPU，不能用于启动准入。
完整模型参数覆盖最终由 `Stage.load()` 校验，数学正确性仍需要独立验收。

启动顺序是 tail 到 head，最后协调器双向认证并完成绑定 nonce/job/ring 的真实请求。
签名者、连续层覆盖和承诺链全部通过后才返回 READY。LISTEN、TCP 成功和 `SHARD_COORD_READY`
仅是启动里程碑。失败清理仅针对本次启动的已登记进程，不按 GPU 型号或端口批量杀进程。

## 3. 同公网 IP 与 SSH 跳板

各容器的 `172.17.0.2` 属于自己的网络命名空间；同出口 IP 不保证互通。下面三种端点
必须分开记录：真实 listener、前驱节点视角的 next、协调器视角的 head/tail。
透明 SSH 转发仍让 strict 握手抵达实际签名节点，不需要降级 legacy。
旧 `--ret-relay` 的内部协议桥接不是透明隧道，仅属于兼容路径。

对全部 caller-local loopback 端点，可以生成跳板路线并在运行前静态核对：

```bash
python phase0/v4_deploy.py tunnels --plan plan.json --config v4-deploy.json \
  --hub N0 --out tunnels.json
python phase0/v4_deploy.py start --plan plan.json --config v4-deploy.json \
  --routes-file tunnels.json --startup-timeout 900
```

把输出的 `routes` 放入配置 `tunnel_routes`。工具在每个 caller 节点建立转发，
不会把所有 `-L` 错建在控制机。caller 到 target/hub 的 SSH 授权应已存在；
控制机的 key 不自动复制到矿工容器。端口冲突、漏掉 return、转发到错误节点会被拒绝。
直连方案可以不配置隧道，最终仍由实际握手/warmup 校验路线和身份。

## 4. 故障证据与生效配置

stage 日志输出 `SHARD_STAGE_EVENT`：starting、loaded、listening、successor/predecessor
connected、failed、stopped。记录 stage/PID/操作、节点本地时间及失败类型。
`failed` 会注明 forward、logits、draft 或 send/receive 操作；完整 traceback 留在日志。
health 汇总各阶段的 owned PID、监听、连接事件和本地首条异常。不同节点的时钟不能
用来证明全局首因；应优先检查计算异常，再核对下游 EOF 的因果关系。

通用 transport EOF 记录 peer 与读取进度，不再把任意对端退出写成 sidecar 故障。
协调器的单作业异常保留 traceback 和结构化 FATAL，后续合法请求仍可被处理。

正常 V4 收据可包含 `shard-runtime-observation/1`：实际进程 run ID/GPU UUID、
源码清单、运行配置摘要、public environment、requested/parsed/observed/verdict、
实际 kernel/Hadamard/graph/wire 后端及版本。该字段进入签名前像；源码改变会拒绝继续
宣称原进程身份。尚未导入的 lazy 模块明确为 UNJUDGED，不把缺省猜成已生效。
未知环境项的值不进入公开观察。

Hadamard 身份另保留实际函数、依赖版本及模块文件摘要，防止只换扩展二进制却声称
同一后端。部署升级应使用新的 checkout/release 路径，排空旧请求后启动新进程；
不在正在服务的 checkout 上原地替换源码并继续使用旧校准。

这些数据仍是节点的签名声明，不是远程硬件证明或模型算术证明；边界见
[V4_TRUST_BOUNDARIES](V4_TRUST_BOUNDARIES.md)。

## 5. 固定基准与单变量比较

按 [V4_BENCHMARK](V4_BENCHMARK.md) prepare/run/verify/compare，保留完整原始记录。
新的协议要求实际 coordinator counters 与签名 runtime observations；旧报告缺项会明确
标记缺失，不能借历史 headline 补齐。准备硬件清单时，从实际 stage 启动证据记录
process_run_id、GPU UUID 和生效环境，不手填未来进程的随机 ID。

兼容 CLI 的 `tokPerSec` 保留原计时口径并输出定义，新的 `decodeTokPerSec` 和
`endToEndTokPerSec` 分离 receipt sweep。报告保留 first/last commit、排空、sweep、
总服务时间以及 P50/P95；投机统计记录真正提交的预测、实际 enqueue/send、作废/排空/未发帧。

`g_cycle` 排除 prefill 的首个 token，`g_frame` 的单位按流水线单 token 帧或串行验证 chunk
分别注明。旧 `g` 仍可能是 generated/(cancels+1)，不可当接受率。
时间加权占用由未舍入的 level/duration 积分计算；该 level 可以包含队列中的工作，
不能冒充实际网络在途包数量。stage recv 包含上游等待，也不能直接命名为 wire RTT。

比较默认要求相同条件，允许明确声明一个变化：

```bash
python phase0/v4_benchmark.py compare before.json after.json --vary source
python phase0/v4_benchmark.py compare before.json after.json --vary env:V4_REFILL_FLOOR
```

实际模型、tokenizer、prompt IDs、输出长度、GPU/driver、层分配及路线必须冻结。
源码实验仍先过数值对照；多个因素同时变化的结果不能登记为单变量收益。
网络声明由 `prepare --network network.json` 冻结，包含 transport、实际路线身份摘要、
测量方法和延迟语义；没有原历史 prompt IDs 的 Rust 同类题只能作为新基线。

## 6. 独立链路测量

[v4_link_probe.py](../phase0/v4_link_probe.py) 在独立端口测试经过实际路线的 32KiB echo。
listener 只绑定 loopback，可通过单独 SSH 转发到达；不使用活跃 stage 端口。

```bash
# 在目标节点，使用独立实验进程/终端。
python phase0/v4_link_probe.py listen --port 29790
# 在 caller，先把本地 30790 转发到目标的 127.0.0.1:29790。
python phase0/v4_link_probe.py run --endpoint 127.0.0.1:30790 --route-id caller-to-tail \
  --transport ssh-via-hub --samples 20 --warmup 3 --out link.json
```

输出全部原始样本、精确收发字节、connect、P50/P95 和有效往返字节速率。
每次 echo 都核对新的 challenge 与完整 payload。该时间包含 SSH/OS/echo 处理，
不是光纤 RTT 的独立测量，也不计入模型推理验收。
