# 本机多卡、专家冷补集与 MTP 图验收

更新：2026-10-29。所有命令供后续独占硬件上的操作员执行。目前完成本地实现与测试；按用户新安排发布默认关闭的代码，真机资格仍待验证。各实验的第一次失败和部分证据必须保留。

本地前置检查已完成：148 个测试文件，**2446 passed / 11 skipped / 97 GPU deselected**（Windows / CPU，795.90 秒）。源码、测试和验收工具在该次运行前后保持一致；1 条 Windows ACL 子进程 UTF-8 解码警告未导致断言失败。以下 CUDA IPC、原生 FP4、整链图、四模式以及 4/6 卡性能验收仍待真机执行。

## 1. 冻结来源，先跑原路径

按[5090 完整验收指南](V4_5090_ACCEPTANCE.md)准备独立目录、源码包 SHA256、原生 checkpoint catalogue、实际 tokenizer、驱动/Torch/TileLang 版本和每张 GPU 的 UUID、PCIe 拓扑、空闲显存。八卡机器不代表八张卡都已授权本轮使用；不能抢占其他测试或清理他人的进程和 `/dev/shm`。

```powershell
.\.venv\Scripts\python.exe tools/v4_acceptance_bundle.py bundle --out C:/Users/Fei/Desktop/FlyAI-dualgpu-20261029-source.tgz
.\.venv\Scripts\python.exe tools/v4_acceptance_bundle.py verify --archive C:/Users/Fei/Desktop/FlyAI-dualgpu-20261029-source.tgz
Get-FileHash C:/Users/Fei/Desktop/FlyAI-dualgpu-20261029-source.tgz -Algorithm SHA256
```

在隔离 Linux 环境先不启用本轮新变量，按原 `resident-whole` 配方完成 whole/eager 回滚、四模式短/长矩阵、strict 同机 4/6 卡和 soak。基线仍需要 `V4_CUDA_GRAPH=whole V4_MOE_IN_GRAPH=1` 等原冻结配方；“零新 flag”不是关闭已有快路径。原生 batch `/2` 另走[scalar/eager/partial 对照](V4_BATCH_GRAPH_ACCEPTANCE.md)。

```bash
python tools/v4_5090_acceptance.py plan \
  --config /data/flyai-acceptance/config.json \
  --out /data/flyai-acceptance/baseline-plan.json
python tools/v4_5090_acceptance.py run \
  --plan /data/flyai-acceptance/baseline-plan.json \
  --run-dir /data/flyai-evidence/baseline-20261029-01 \
  --confirm-gpus 0,1,2,3,4,5,6,7 --execute
python tools/v4_5090_acceptance.py verify \
  --plan /data/flyai-acceptance/baseline-plan.json \
  --run-dir /data/flyai-evidence/baseline-20261029-01
```

示例 GPU 列表必须与独立冻结计划完全一致。该流程没有自动为本轮每个 opt-in 组合颁发资格；改任一变量、工件或传输配置后应新建校准、计划及输出目录。

## 2. 最小本机 CUDA 专项

先确认所选两张真实 GPU 均空闲；下面把物理 GPU 6、7 映射为本进程 cuda:0/1，仅是例子。CUDA 专项和原环应顺序执行，避免争用。Windows CPU SHM 测试不能替代 Linux CUDA IPC。

```bash
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=6,7 python -m pytest \
  tests/test_v4_host_transport.py tests/test_v4_expert_tiers.py \
  tests/test_v4_draft_chain.py -m gpu -q \
  --junitxml /data/flyai-evidence/dualgpu-cuda.junit.xml

CUDA_VISIBLE_DEVICES=6,7 python phase0/v4_host_transport_probe.py \
  --source-device cuda:0 --target-device cuda:1 --execute-gpu \
  --tokens 32 --dim 4096 --hc 4 --max-mib 16 --repeats 20 \
  --out /data/flyai-evidence/activation-copy-probe.json
```

IPC probe 必须有实际 `gpu_copy_observed=true`、相等的真实字节 hash、干净进程退出和零隔离槽遗留。它是合成激活、没有资源租约或 strict 会话的探针，`performance_qualified=false`，不能直接喂给 planner 当链路校准。

exclusive 的 CUDA case 检查 packed FP4 数据和 scales 的迁移字节，以及多轮 D2H/H2D；它不运行整个真实模型。MTP 的 CUDA case 是真实 capture/replay 加 native attention/Head、合成 dense FFN，不能视为原模型 packed MoE 完整图通过。原模型资格还需要下一节的独立对照。

`phase0/v4_release.py gpu-check` 及其 frozen test inventory 仍须按当前 CLI 和 release 合同运行；一个真实两卡 test 不能在仅可见一张 GPU 的进程内通过。skip 只能登记未验证，不得删除必需 case、放宽 gate 或伪造 CUDA 可用性。

## 3. 单变量原模型对照

每个实验固定 prompt/token IDs、checkpoint、量化布局、同一层段、输入、上下文、wire、依赖与采样策略，先 cold 请求再相同 warm 重复。分别记录 TTFT、warm decode、端到端速率、轮时、g、commit token、逐卡峰值和实际快路径计数。不要把 TTFT 直接解释为编译时间，或从每 token 总时长倒推出“全是 host”。

| 实验 | 原模型必须核对的内容 |
|---|---|
| 角色成本 | head、中段、tail/MTP 分别五次以上 completion-fenced 样本；同模式/角色/源码/形状绑定；planner 保留固定 Head/MTP 成本，不只比较层数。 |
| 本机激活 | 原 socket 与私有 CUDA IPC 各一条 strict 同环；相同提交 tokens、相邻真实字节 roots 和 signer；实际 IPC/fallback/ACK、driver 显存与 SHM 预算。 |
| exclusive | graph-off 的 canonical RAM cache 作为对应基线；相同容量、专家/scale 字节、层输出及可读 KV；热冷置换超过缓存容量；保存/重载、取消、失败、读者排空后重用。 |
| 完整 MTP 图 | 同一个原生 checkpoint 分别建立独立原 drafter、FAST eager 与 chain drafter；按已提交输入逐轮比较 drafts、logits、confidence 及所有可读 MTP KV 字节；不能只比 argmax。 |
| 宿主预算 | 同一 DB、相同 budget_id 的两个 worker 争用，超额明确拒绝；真实 copy 完成前不能再占槽；过期活动许可不释放；控制线程保留、线程数和可选 affinity 实测。 |
| draft 适配器 | 原 teacher 数据与独立对话留出；目标模型参数摘要不变；最差草稿/拒绝恢复仍发出相同 verifier tokens；中文/代码/散文/工具任务的接受率、轮时和独立质量。 |

`phase0/v4_draft_chain_probe.py` 已提供原模型独立对照入口。它串行启动两个隔离子进程，分别加载一个相同原生尾段与 MTP，运行原 scalar drafter 和完整 chain，不要求显存同时容纳两个模型。冻结输入 safetensors 包含 `prefill_ids [1] int64`、`prompt_taps [1,P,targets*dim] bf16`、`committed_ids [1,N] int64`、`main_taps [1,N,targets*dim] bf16`、`run_lengths [R] int64`；各 run 之和为 N，且至少两轮。使用真实目标执行导出的配对 token/taps，保存原始数据来源。

```bash
CUDA_VISIBLE_DEVICES=7 python phase0/v4_draft_chain_probe.py \
  --checkpoint "$CHECKPOINT" --input "$INPUTS/frozen-chain.safetensors" \
  --out-dir "$EVIDENCE/native-chain-01" --lo 40 --hi 43 \
  --execute-gpu --device cuda:0 --host-mib "$HOST_MIB" \
  --device-mib "$DEVICE_MIB" --graph-mib 1024 --max-seq 8192
```

此处 1024MiB 只是预算示例。每臂输出完整 `scalar/chain.safetensors`、逐 tensor 字节摘要、配置、后端、硬件身份、计数和日志；总报告在 `report.json`。GPU 必须真实捕获且 `replays>captures`、零失败/回退、原生 TileLang 与硬件身份齐全、逐轮 drafts/logits/confidence/完整 KV 零容差一致；CPU fixture 的 `gpu_qualified` 始终为 false，单次通过也不构成整个产品的 release qualification。

原模型 campaign 还须覆盖 run 长度 `1..block_size+1`、window 未饱和与绕回、reset 后的新 token/hidden、捕获失败恢复、图数和字节上限、输入／权重／rope 变化拒绝。探针两遍 reset/replay 复用相同冻结输入；换输入另跑新目录。tiny CUDA test 和单个 trace 不完成全部原模型 obligation；缺任一项写 pending，不移用其他路径报告。

四模式 greedy、spec、dspark、pipelined 仍逐项核验 reset/cancel/reconnect/recovery；MTP 图只对实际使用草稿的模式计覆盖。ngram 和 greedy 不应伪报 MTP capture。原 whole 回滚与 batch 图的资格彼此独立。

本指南的原 exclusive 基线采用 `V4_CUDA_GRAPH=0 V4_MOE_IN_GRAPH=0 V4_DSPARK_MOE=0 V4_KV_PLACEMENT=gpu`、固定 `V4_EXPERT_CACHE_SLOTS` 和完整 host 预算。最新显式分段图例外需要 whole-eager、正的 `V4_EXPERT_SEGMENT_GRAPH_MIB` 和 resident KV，按[六项新验收](V4_SIX_ACCEPTANCE.md)单独测量；不能只删除拒绝条件。完整 MTP 图仍采用 resident experts，不能同时打开 exclusive 或 RAM expert path。

完整 MTP 图配置示意，容量按实测更换，须在 import 前注入：

```bash
export V4_EXPERT_PLACEMENT=gpu V4_EXPERT_STORAGE=canonical
export V4_EXPERT_STORAGE_HOST_MIB=0
export V4_DSPARK_FAST=1 V4_DSPARK_MOE=1
export V4_DSPARK_CHAIN_GRAPH=1 V4_DSPARK_CHAIN_MIB=512
export V4_DSPARK_CHAIN_LIMIT=16 V4_DSPARK_CHAIN_REFERENCE=0
```

512MiB 是显式预算示例，不保证任何 checkpoint/context 能容纳；备份、static inputs/outputs、私有 graph pool 和失败后 allocator 保留均须计费。输入变化必须真实刷新，不能重复同一输出冒充 replay。

## 4. 离线草稿数据、训练和词表审计

先由原模型 teacher-forced 导出至少四段独立、已获使用授权的对话，一文件一对话；不要自动收集服务中的客户提示。每个 safetensors 的字段是：

| 字段 | 形状及含义 |
|---|---|
| `prompt_taps` | CPU、native dtype，`[1,P,targets*dim]`，完整 prompt 的原 target-layer taps。 |
| `anchor_ids` | CPU int64 `[N]`，每个 main tap 对应位置 p 的已提交 anchor token p+1。 |
| `main_taps` | CPU、native dtype，`[1,N,targets*dim]`，连续位置 p 的真实 main taps。 |
| `teacher_logits` | CPU float32 `[N,native_vocab]`，teacher 对 p+2 的原词表分布；不得用被拒绝草稿后缀冒充。 |

每段 N>=2，P>=1，预留 native MTP horizon；CLI 总输入和输出有明确 host bound。先核对 teacher/tokenizer 对齐与时间偏移，再运行 collector；工具能约束结构和身份，不能证明外部教师数据执行诚实。

```bash
CUDA_VISIBLE_DEVICES=7 V4_CUDA_GRAPH=0 V4_NATIVE_LANES=0 \
V4_NATIVE_BATCH=0 V4_EXPERT_PLACEMENT=gpu V4_DSPARK_CHAIN_GRAPH=0 \
python phase0/v4_draft_training.py collect \
  --checkpoint "$CHECKPOINT" --lo 40 --hi 43 --device cuda:0 \
  --inputs "$INPUTS/conv-1.safetensors" "$INPUTS/conv-2.safetensors" \
           "$INPUTS/conv-3.safetensors" "$INPUTS/conv-4.safetensors" \
  --out-dir "$EVIDENCE/draft-data" --max-tokens 8192 \
  --host-mib "$HOST_MIB" --device-mib "$DEVICE_MIB"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase0/v4_draft_training.py train \
  --input "$EVIDENCE/draft-data/training.safetensors" \
  --binding "$EVIDENCE/draft-data/binding.json" \
  --out-dir "$EVIDENCE/draft-adapter" \
  --host-mib "$TRAIN_HOST_MIB" --rank 8 --steps 100 --batch-size 16 --lr 0.01

python phase0/v4_draft_training.py audit \
  --decoded-tokens "$INPUTS/native-decoded-tokens.json" \
  --proposed-ids "$INPUTS/native-draft-token-ids.json" \
  --target-ids "$INPUTS/heldout-target-token-ids.json" \
  --out "$EVIDENCE/vocabulary-audit.json"
```

collector 必须在无旧 adapter、无完整草稿图、无 service lanes 的独立进程运行；其默认 `--device=cpu` 不是 CUDA 测量，真实原生 FP4 采集要显式给设备。它采集实际 MTP slot 0 特征和原始 logits，训练用原模型完整词表的低秩 residual，不采用 Strata 模型的 143834 子词表。

训练报告按 group 整组划分，保留数据、binding、adapter、manifest、training report 和独立 SHA256。单 token 解码脚本分类只是 CJK/Latin 覆盖下界，字节碎片可能归 other；不能把覆盖审计当中文质量。留出损失退化的工件不会安装；即使改善也没有质量或加速资格。

在新建进程、任何草稿图 capture 之前，以 `V4_DRAFT_ADAPTER_DIR=<私有工件目录> V4_DRAFT_ADAPTER_MIB=<显式容量>` 加载一次，重新冻结 runtime/cohort 和资源测量。实测目标模型权重 hash、verifier tokens 与恢复状态；适配器绑定失配、摘要变更、显存不足、非有限值必须拒绝。不要覆盖已有图内权重或共享 target Head。

## 5. 发布与归档

逐项通过后再试代码允许的组合，每组合单独冻结计划。检查 signed runtime 的静态配置与 stage-lifetime `local_execution` 观察；累计计数不能直接当某个请求的命中率或图次数。共享内存 witness、CUDA driver allocation、cold/exchange banks、graph 和 adapter 预算不能漏报。

保留源码包／HEAD／环境清单／实际配置摘要、frozen inputs、严格签名收据、JUnit、原始日志、首个偏差、内存峰值、生命周期和清理结果。不要归档私钥、secret、私有路径或未经授权的提示。先复制到持久磁盘并回收到本地，再退租。

`phase0/v4_release.py freeze/verify` 保留四卡、六卡、活跃 soak 和真实 CUDA gate 的完整要求；新专项不能代替它。四卡 >=40、六卡 >=30 committed decode tok/s 仍以完整套件 warm 中位数与正确性一起判断。任何 skip、fake clock、模拟器或 CPU fixture 都只能登记相应未验证项。源码发布不替代真机验收；推荐配置须经维护者审核，脚本不会自动操作 Git。
