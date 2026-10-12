# V4 多卡宿主执行与本地验收

更新：2026-10-11。本轮代码在本地工作树，GPU 验证和 GitHub 推送由用户留待后续。
下述命名配方、探针和 CPU/socket 测试不是新的 5090、4090 或跨公网速度成绩。

## 1. 实验记录与验收对象

OpenClaw 本次归档的六段 localhost legacy greedy 第二轮为 192 个输出 token：
decode 21.093 tok/s、端到端 17.125 tok/s、首字 2.156s。同六段 island 第二轮为
3.430 decode tok/s，约为 6.15 倍差异。第一轮与第二轮必须分别保留，不能估算成
“含预热 15–20 tok/s”，也不能把 21.09 写为落在 24–28 区间内。

两组相同明文提示在第 5 个 token 分歧：island=23205、whole=24414。
共同开启 REF_SLIM，且没有完整输入 token、源码/权重身份和原始签名收据。
因此这是执行路径线索，不是保持原输出的性能验收，也没有排除其它网络的瓶颈。

whole 的第一层数值验收对象是 capture-safe eager twin；它与旧参考路径的
Indexer 排序及 slim 覆盖边界另行比较。本轮修复拒绝后的 whole 回滚，使其使用
同一个 eager twin，避免正常执行和重放进入不同的短上下文数值路径。
新诊断也不放宽原来的提交 token 一致性要求。

## 2. 冻结输入并定位分歧

`phase0/v4_numeric_diagnose.py` 每次 trace 在独立进程执行，防止两个配方的
import-time 常量、模型函数替换和默认 dtype 互相污染。输入是 safetensors：
`token_ids` 为 CPU int64 `[1,total]`；非 head 还需对应的
`hidden [1,total,hc_mult,dim]`。它不调用 tokenizer，也不启动环。

```bash
python phase0/v4_numeric_diagnose.py trace \
  --checkpoint /data/v4-stage --input frozen-input.safetensors \
  --prompt-tokens 32 --lo 0 --hi 8 --device cuda:0 \
  --execution configured --env-file whole.json --out whole-trace.json
python phase0/v4_numeric_diagnose.py trace \
  --checkpoint /data/v4-stage --input frozen-input.safetensors \
  --prompt-tokens 32 --lo 0 --hi 8 --device cuda:0 \
  --execution whole_eager --env-file whole.json --out twin-trace.json
python phase0/v4_numeric_diagnose.py compare \
  whole-trace.json twin-trace.json --out numeric-diff.json
```

层范围和输入仅为示例，须换成同一已校验 stage 的实际范围。生产 trace 要求真实
verified artifact witness；`--fixture` 是独立 CPU 测试范围。默认 token/记录/张量
预算有界；工具输出逐层与有效 KV/压缩状态哈希，报告第一处分歧，而非保存整份 KV。
`--rejection` 可以显式安排位置、草稿宽度和接受前缀，核验回滚后的状态和后续输出。
`reference`、`whole_eager`、`configured` 三条路径分开记录，twin 不被统计为实际图重放。
示例 `whole.json` 可以仅包含 `{"V4_EXECUTION_PROFILE":"resident-whole"}`；数值诊断只
接受它列出的数值开关，不能将含私密路径或任意服务配置的完整环境文件直接传入。

## 3. 一个配方来源，多种明确合同

冻结 benchmark 默认环境与 `v4_execution_profiles.py` 共用同一个 baseline。
命名配方只有显式选择时生效，裸启动的兼容默认保留。

| 配方 | 图与专家 | 数值限制 |
|---|---|---|
| `resident-whole` | whole，GPU 常驻，Grouped MoE 入图 | REF_SLIM、NOQAT、FAST_VERIFY、FP8 wire 关闭 |
| `resident-island` | island，GPU 常驻，MoE 不入图 | 相同受约束开关，供受控对照 |
| `ram-experts` | whole attention 前后图，RAM 专家动态 seam | MoE/MTP 优化分派不入图，实际缓存与共享内存预算必需 |

在新进程中设置 `V4_EXECUTION_PROFILE`，必须先于 runtime 模块导入。展开后可以
调整上下文和资源预算；不得用同一个配方名字打开改变其数值/放置合同的选项。
完全晚注入已展开的环境也不能为之前读取了旧常量的模块补资格。
`V4_CHUNK_PREFILL`、`V4_PREFETCH` 被拒绝，并指向实际变量名，不自动猜测替换。

操作方可用 `resolve_profile(name, overrides)` 生成完整 JSON 环境。严格本机模板和
校准须绑定展开后的环境，而不是只绑定一个名字。手工部署、leased runner 和
Stage 使用同一来源；私钥仍通过原私密通道配置，不能写进公开环境或日志。

named-profile warmup 在签名/身份/覆盖校验后检查每个实际层的 captured/eager 状态。
island 必须保有三个真实图；whole 常驻配方还需实际 MoE 入图；RAM 配方允许动态图外
MoE，但不能将完全 eager 冒充 whole。普通和 native-lane warmup 使用同一个门槛。
两 token smoke 不代表所有 bucket、压缩边界、长上下文和速度线已经通过。

## 4. 加载大权重前检查原生内核

```bash
python phase0/v4_kernel_capability.py --device cuda:0 --graph --timeout-s 900
```

该显式工具在独立、可回收的子进程使用有界 synthetic 原生格式权重，执行 FP4
量化、FP8/FP4 GEMM、六个激活专家的 Grouped MoE、attention 及变更输入后的图重放。
报告绑定 UUID、计算能力、驱动、CUDA、Torch、TileLang 和源码身份。缓存只保存历史，
不会跳过本次执行。超时回收自己启动的 worker 与编译子进程，不按名字通杀其它任务。

preflight 可选配置：

```json
{"native_kernel_probe":{"enabled":true,"graph":true,"timeout_s":900}}
```

默认预检不执行新增 GPU 计算。CPU、缺依赖、无法确认清理和不支持的编译/ISA 路径
各自明确分类；没有 BF16 隐式降级。探针通过只代表所列形状的 kernel smoke，不能
代替原生整层、真实权重 stage、资源峰值和完整环验收。Torch allocator 上限不包含
driver context 和编译器的主机内存。

## 5. 宿主分组、拓扑和本机传输

同宿主仍是多个独立 GPU worker：各自 UUID、签名、实际跨度与租约，共享宿主 RAM、
pinned RAM、磁盘和 I/O 预算。同公网 IP 不用于推断宿主身份。

在原 strict formation 的 `locality` 中显式启用：

```json
{"host_groups":{"enabled":true,"mode":"contiguous",
  "topology_reports":[],"transport_samples":[]}}
```

规划只使用真实校准的 allowed spans，尽量保持宿主连续层段，并保留每 GPU 的
严格会话和真正 tail 直接返回；不会调用旧 legacy 均分助手。减少卡数首先通过
单卡峰值检查，不能靠总显存相加或 `expandable_segments` 绕过预算。

```bash
python -m shard.host_probe --devices 0,1,2,3 --host-id rig-a --dir /data/v4 \
  --out host-io.json --topology-out host-topology.json \
  --pair-sample-mib 1 --pair-repeats 10 --max-pairs 64
```

GPU-pair copy、peer capability 和 Linux PCI/NUMA 只读观察是诊断，不证明当前引擎
已经使用 CUDA IPC，也不自动绑定 CPU 或内存亲和。`device_copy_not_engine_wire`
不能用于替换真实 socket 边成本。

只有 `shard-host-transport/1` 的真实 engine ready-to-ready 样本，准确匹配两端
UUID、完整源码摘要、runtime 配置、cohort、形状、实际 route/channel 和 TTL，才能
降低预测成本。两端 stage trace 必须明确只含 receive 后、send 前的计算，防止双计。
缺任一绑定保留原估算和 uncertainty；平均值不自动构成 P95 SLO。

可选接收缓冲：

```bash
export V4_TRANSPORT_BUFFER_MIB=8
export V4_TRANSPORT_BUFFER_SLOTS=2
export V4_TRANSPORT_PINNED=0
```

缓冲池复用 frame storage，避免一次 CPU blob 副本；最后一个 tensor view 持有槽位。
池占用、resize 临时峰值和共享 host/pinned 预算有界；空间或可选分配失败时走原 owned
decoder。帧格式、签名和 owner fences 保留。默认关闭，接收池统计是进程累计值。
当前 H2D 是阻塞路径；未来若引入异步 H2D，必须在 CUDA event 完成后才能释放借用槽位。

诊断普通与 buffered loopback（CPU 也可执行）：

```bash
python phase0/v4_local_transport_bench.py --tokens 1 --repeats 20 \
  --buffer-mib 8 --out local-wire.json
```

显式指定 `--source-device cuda:0 --target-device cuda:1 --pinned` 才会使用 GPU。
这份 synthetic、无租约、无 strict envelope 的报告永远不是 planner engine 样本。
该优化仍走 TCP 和 D2H/H2D，不能标成 GPU 零拷贝或 CUDA IPC。

## 6. 四策略公平比较与独立故障矩阵

正式 benchmark 支持 `greedy / spec / dspark / pipelined`；`spec` 是协调器 n-gram，
`--ngram-k`、`--ngram-order` 被冻结，未冻结的 n-gram margin override 被拒绝。
`compare --vary mode` 只允许策略和其 n-gram 参数变化，其余环境、源码、硬件、
网络和输入仍固定，跨报告 greedy 输出必须相同。`g` 不替代实际提交速度。

```bash
python phase0/v4_benchmark.py prepare --dir /data/v4-assets --hardware hardware.json \
  --isolation none --env-file recipe.json --network network.json \
  --mode spec --ngram-k 4 --ngram-order 3 --out spec-protocol.json
python phase0/v4_benchmark.py compare greedy-report.json spec-report.json \
  --vary mode
```

prepare 之后仍需原 run/verify 步骤，详见 [V4_BENCHMARK](V4_BENCHMARK.md)。
故障注入单独运行，不计入 tok/s：

```bash
python phase0/v4_soak.py lifecycle --protocol protocol.json --dir /data/v4-assets \
  --head 127.0.0.1:29610 --tail 127.0.0.1:29615 --deployment-plan plan.json \
  --coordinator-key /run/private/controller.key --cycles 3 --out lifecycle.json
```

该矩阵重复 reset、中途取消、新 owner fence 重连、恢复原输出，并用明确错误草稿
触发拒绝回滚，保留签名 grants 与完整作业 receipts。它使用既有分配的严格环；
不会租卡、启动节点或降低认证。原有持续 soak 与发布门槛继续保留。

## 7. 本地验证与硬件边界

本轮真实 CPU 数学、strict TCP、签名链、资源规划与进程树测试可在本机执行。
所有新增 CUDA、pinned allocator、GPU-pair 和原生 kernel 测试保留独立 GPU 标记；
未执行不算通过。组合回归的实际数量和环境跳过在工作树 JUnit 中记录。

本轮 Windows/Python 3.11 CPU 组合：**123 文件，2081 passed、11 skipped、80 GPU deselected**。
工作树证据为 `.venv/flyai-multigpu-regression-final-20261011.junit.xml`，文件清单为
`.venv/flyai-multigpu-regression-files.json`；专项不累加。首次组合检出了 CLI/runtime
同名与 flat/package transport 配置分叉，修复后完整重跑上述范围。语法、diff 和文档
引用检查通过；没有把本轮登记为全仓跨平台 CI 或 GPU 性能通过。

`v4_release.py gpu-check` 已纳入本轮 whole 回滚、island 覆盖、原生 kernel 和 pinned
缓冲 CUDA 门槛，且使用独立 owned 进程树监督。CPU/busy GPU 不执行该门槛；清理未确认
或任一必需 CUDA 用例缺失/跳过时不通过。Windows 清理有真实父子进程回归，POSIX
对应清理分支仍需后续 Linux 实测。

后续硬件顺序：单张 5090 原生 kernel/数值/回滚 → 实际六卡 greedy 配方 → 四策略
同口径比较 → 长上下文及持续稳定性 → 宿主间网络对照。4090 应先独立过量化内核
探针，再谈适配；便宜卡的控制面测试不能替代 5090 性能资格。真机验证后再推送。
