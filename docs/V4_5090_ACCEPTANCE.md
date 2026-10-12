# 5090 真机验收：whole 回滚、同机多卡与四种解码

更新：2026-10-12。执行架构代码基线为 `01e4c87`；验收工具随源码发布，本地回归通过，真机验收待执行。
交给 OpenClaw 的应是核对哈希的源码包，或包含本轮验收工具的固定 Git 提交；不要使用缺少工具的旧版本。

验收报告中的 `github_push_allowed` 记录实验是否达到全部验收门槛；脚本不会执行 Git 操作。
源码提交按维护者指令进行，GPU 性能资格仍由真机证据确认。

## 验收范围和通过条件

默认在一台 8×RTX5090 上用 GPU 0–5 建六段环，GPU 6 做真实权重数值诊断，GPU 7 跑 CUDA 专项。
每个 stage 是独立进程、独立签名身份、独立 GPU UUID；层范围必须连续且覆盖 43 层。
示例 `[0,7,15,23,31,39,43]` 给首卡嵌入和尾卡 MTP 留出空间，仍须逐卡真实测量。
不能用总显存或平均权重大小推断能装下。示例保留 2GiB 显存余量；不足即停，不自动关闭优化或改成 RAM 专家。
本轮只验证原生 DeepSeek-V4-Flash 工件；不切第二模型，不引入跨节点专家副本。

| 门槛 | 必须提供的证据 |
| --- | --- |
| CUDA 专项 | 实际 SM120 GPU 运行的全部必需测试，原始 JUnit；skip、CPU fixture 不算通过 |
| whole 数值/回滚 | 真实权重最短 head 前缀（当前配置为 0:4），逐层输出及有效 KV/压缩状态精确匹配 whole_eager；拒绝、恢复及两个压缩比、Indexer 桶切换均被覆盖；实际捕获且无回落 |
| 同机多卡 | 逐角色资源探针、真实 UUID、精确层分配、strict 签名收据、同一个 host_id、每段 whole+MoE 实际捕获 |
| 四模式 | greedy、ngram spec、MTP dspark、pipelined：短/长两个上下文，每种包含 code/math/prose/agentic 四类固定输入、三次 warm、greedy 对照；提交 token 必须一致 |
| 性能 | 至少一种短上下文模式的**完整套件 warm decode 中位数**达到六卡 30 tok/s（四卡配置为 40）；其余模式可以较慢，但正确性、签名和捕获必须有效 |
| 可靠性 | 四模式 reset/cancel/reconnect/recovery，强制错误草稿后恢复；至少一小时**活跃** soak，按独立冻结的 TTFT/token-gap P95 条件验收；完整受管进程清理 |

`valid_baseline` 表示证据有效但未达速度线，不能写成验收通过。`V4_CUDA_GRAPH=whole` 日志或高 tok/s 都不能替代实际捕获与输出验证。
数值专项是**局部 stage**的精确证明；四模式环测试是提交 token 对照，不能宣称全模型隐藏状态逐位一致。
这是同机六卡的操作员证据，不是远程硬件证明，也不等于 `v4_release.py` 要求四卡、六卡同时达线的完整发布证书。
localhost 结果不能用来断言 WAN 不再是瓶颈，5090 通过也不代表 4090 已通过。

## 1. 出发前：冻结待测源码

在本地工作目录执行（PowerShell）：

```powershell
.\.venv\Scripts\python.exe tools/v4_acceptance_bundle.py bundle --out C:/Users/Fei/Desktop/FlyAI-5090-acceptance-20261012-source.tgz
.\.venv\Scripts\python.exe tools/v4_acceptance_bundle.py verify --archive C:/Users/Fei/Desktop/FlyAI-5090-acceptance-20261012-source.tgz
Get-FileHash C:/Users/Fei/Desktop/FlyAI-5090-acceptance-20261012-source.tgz -Algorithm SHA256
```

源码包保留当前文件字节、基础 Git HEAD、核心源码和验收工具哈希；不携带 `.git`、权重、私钥、虚拟环境和历史 GPU 成绩。
源码包哈希要在本地另行保存，传到服务器后先核对，避免拿错版本。
后续追加代码后须重新生成包，不能继续用旧哈希。

## 2. 租到机器后：独立目录和依赖

先确认 0–7 全部属于本轮且空闲。有人占卡时停止，不能杀其他进程、改它的目录或抢卡。
SSH 地址和凭证由租赁后提供；工具不自动 SSH、下载权重、装包、改驱动或改系统设置。

把包上传到 `/data/flyai-acceptance/`，在 Linux 上：

```bash
cd /data/flyai-acceptance
sha256sum FlyAI-5090-acceptance-20261012-source.tgz
# 和本地保存值一致后，再解压到新的隔离目录；不要覆盖 /root/FlyAI。
mkdir -p source
tar -xzf FlyAI-5090-acceptance-20261012-source.tgz -C source
cd source/FlyAI
python3 -I tools/v4_acceptance_bundle.py verify --archive ../../FlyAI-5090-acceptance-20261012-source.tgz
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
```

用独立虚拟环境准备仓库要求的 Torch/CUDA、TileLang、safetensors、tokenizer、psutil、pytest 等依赖，参见
[部署指南](V4_CLUSTER_DEPLOY_GUIDE.md)与 [原生内核探针](V4_MULTIGPU_EXECUTION.md)。
首次编译需要可用 nvcc，Torch 的 CUDA、编译器与 SM120 支持须一致；不要根据 PATH 中某个 nvcc 的版本猜测实际后端。
Hadamard 是否扩展成功由探针/实际运行报告确认，不把“装上扩展”直接等同于性能提升。

权重目录必须是已转换、带原生 catalogue/pack 元数据和匹配 tokenizer/config 的工件，参见
[权重准备](WEIGHT_PREPARATION.md)。原始 Hugging Face 权重目录不能直接替代。
`catalogue_if_missing=false` 是默认值；若确有完整的已转换权重、只是缺元数据，可显式改为 true，工具会流式校验现有工件，不下载或转换权重。
目录本身或内存盘可能在退租/重启后消失，输入及全部证据应回收到持久磁盘和本地。

## 3. 改配置，然后只规划

复制 [示例配置](../examples/v4_5090_acceptance.json) 到源码目录之外，修改 `model_dir`、`python`，核对 GPU 编号、层边界和端口。
不要把本地 Windows 虚拟环境路径带到 Linux。
示例使用 `resident-whole`：whole、MoE in-graph、无 REF_SLIM/FAST_VERIFY/FP8-wire 近似，max_seq8192、prefill query chunk128。
工具补齐并冻结所有注册的公共环境默认值；拒绝 `V4_CHUNK_PREFILL`、`V4_PREFETCH` 等旧变量名。
不要在 import 后临时注入开关，不要边跑边改配方。

```bash
cp examples/v4_5090_acceptance.json /data/flyai-acceptance/config.json
# 编辑 config.json，确认实际虚拟环境、工件目录及独占 GPU。
/opt/flyai-venv/bin/python tools/v4_5090_acceptance.py plan \
  --config /data/flyai-acceptance/config.json \
  --out /data/flyai-acceptance/plan.json
sha256sum /data/flyai-acceptance/plan.json
```

`plan` 是 CPU 规划，不读模型权重、不创建密钥、不探测或启动 GPU。
计划内容哈希和计划文件的 SHA256 是两个不同字段，都应保存。实际验收不能改用另一个计划。
输出目录必须在源码目录外且不存在，避免污染源码身份，或覆盖前一次证据。

## 4. 正式执行：一条入口，顺序受控

```bash
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/flyai-venv/bin/python tools/v4_5090_acceptance.py run \
  --plan /data/flyai-acceptance/plan.json \
  --run-dir /data/flyai-evidence/run-20261012-01 \
  --confirm-gpus 0,1,2,3,4,5,6,7 --execute
```

没有 `--execute` 时只返回计划状态。`--confirm-gpus` 必须是计划所需的完整独占集合。
入口依次执行 CUDA 专项、独立真实模型数值测试、逐卡资源探针、8 个矩阵单元、全模式生命周期及持续 soak。
四模式×两种长度×每单元 20 个 job，共 **160 个基准 job**，另有数值、编译与一小时以上 soak；预算应按数小时安排。
各阶段有超时和日志大小上限；任何缺证据、OOM、身份差异、错误输出、whole 回落或 cleanup 未确认都会停止并保存部分证据。
不降级到 legacy，不自动扩大显存预算，不以重试成功覆盖第一次失败；修复后用新计划和新目录重跑。

每个矩阵单元都重新加载进程/模型：先仅 loaded、请求数为 0、`ready=false`，冻结实际硬件身份，然后发送首个基准请求。
首个请求之后才做 warm 重复和 greedy 对照。结束再运行签名 warmup，验证 READY，随后停止本轮受管进程。
这测量的是进程/模型冷启动后的请求；CUDA 专项和资源探针可能已预热磁盘 JIT 缓存，**不宣称编译缓存全冷**。
端口可连、`listening` 或 loaded 只说明进程阶段，不能替代签名 READY。

whole 专项使用冻结的真实 tokenizer ID。当前原生配置自动选最短含 ratio4/128 和 Indexer 的 head0:4，覆盖短上下文、压缩边界、
实际第一个 Indexer 桶边界（当前 end4099→4100）以及拒绝后的恢复。输入共4104 token，运行容量8192，输出只保存有界状态哈希。
如生产配置结构改变，脚本须从工件推导边界；覆盖不足只能 partial/unverified，不能用 fixture 或随机隐藏状态补成通过。

## 5. 核验和回收：不相信 summary 中的自报结果

```bash
/opt/flyai-venv/bin/python tools/v4_5090_acceptance.py verify \
  --plan /data/flyai-acceptance/plan.json \
  --run-dir /data/flyai-evidence/run-20261012-01 \
  > /data/flyai-evidence/run-20261012-01.independent-verification.json
sha256sum /data/flyai-evidence/run-20261012-01.evidence.tgz
```

核验重新读取独立计划、原始 trace、输入、协议、签名收据、提交 token 和计时，不采信手改 summary、状态字段或单轮峰值。
自动生成的 `.evidence.tgz` 在成功/失败时均保留已有公开证据，排除签名私钥、私有请求及进程所有权数据库；权重不打包。
将 `.evidence.tgz`、外部 `plan.json`、计划 SHA256、归档 SHA256、核验输出和控制台退出码全部传回本地后才能退租。
归档解压后可使用原源码包在本地重核验；numeric 输入/trace 按归档目录定位，无需重租 GPU或保留权重。
独立 verifier 还要求同一验收工具与核心源码，不能拿后续修改的 checkout 核验旧计划。

| 主要产物 | 用途 |
| --- | --- |
| `campaign-plan.json` / `index.json` | 冻结合同、执行范围、完整性与清理状态 |
| `numeric/` / `numeric-input.safetensors` | 实际 whole/eager、回滚、压缩及桶边界 raw trace 和精确比较 |
| `gpu-gate.json` / `gpu.junit.xml` | 实际原生内核和状态隔离专项 |
| `ring/resource-stage-*.json` / `allocated-gpus.json` | 每 GPU/角色/source/runtime 的显存峰值和真实 UUID |
| `matrix/{short,long}/{mode}/` | 每模式独立协议、签名报告、cold loaded、signed warmup、部署计划 |
| `service/` | 独立 soak/lifecycle 合同及完整原始报告 |
| `logs/` / `stage-logs/` | 受管子命令和逐段原始日志 |
| `summary.json` / `evidence-manifest.json` | 汇总和逐文件哈希；不是无需复核的发布证明 |

若被 Ctrl-C 中断，入口尽力关闭自身记录的 PID/子进程并归档；不能确认时查看 `ring/stop.json`，只恢复本轮受管进程。
不要使用 `pkill python`、清空 `/dev/shm` 或删除他人的权重。强制关机/SIGKILL 可能无法执行 finally，应先保留现场，再按本轮 state 做 owned cleanup。

## 交给龙虾的任务文字

> 用我提供并核对 SHA256 的 FlyAI 本地源码包，在独立目录和独立环境运行 `docs/V4_5090_ACCEPTANCE.md`。
> 先确认计划所列 GPU 全部独占且空闲，核对原生权重 catalogue/config/tokenizer。默认六卡环加两张专项卡；不要抢其他任务。
> 先 plan，再按冻结计划 run --execute。必须完成真实 whole/eager 回滚/状态哈希和压缩/桶切换、逐 GPU 资源及签名捕获、
> greedy/spec/dspark/pipelined 的短/长矩阵、生命周期和一小时活跃 soak。不要混用旧变量名、legacy、近似开关或测量口径。
> 任何失败就保留原始日志和部分证据，不用改变配方或删除第一次失败来凑通过。总结各模式 cold TTFT、warm decode、端到端速率、
> acceptance/在途指标、逐卡显存、实际 whole 捕获和首个不一致位置；缺测明确标未验证。
> 独立 verify 后将源码/计划/证据包 SHA256、原始归档、核验输出及退出码全部回收到本地，完成后再退租。
> 真机任务只回收验收证据；不要把缺测或失败写成已通过的性能资格。新修复由我们复核后另行提交。

本地 CPU/socket 回归只验证工具与合同，不能预先勾选上述 GPU/速度门槛。

## 本地验证记录

2026-10-12：本轮六个验收测试文件以及 soak、benchmark、strategy matrix、release、deploy、managed launch 六个相关文件，
组合运行结果为 **294 passed / 2 skipped / 3 GPU deselected**，用时124.67秒。
两个 skip 分别是 Windows 不可用的符号链接创建和既有 POSIX 进程组专项。
包含真实 CPU/socket、真实签名、真实受管进程及后代清理测试；不代表全仓 CI 或原生 GPU 已通过。
JUnit 保存于本地 `.venv/v4-5090-acceptance-local-20261012.junit.xml`；源码包排除虚拟环境。
Python 编译、Git diff 空白检查及仓库文档引用检查通过。
