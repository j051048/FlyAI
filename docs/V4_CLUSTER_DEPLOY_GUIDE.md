# FlyAI DeepSeek-V4 多机与多卡集群部署实操指南 (Runbook)

本指南针对在 **Vast.ai、AutoDL 等云端租机平台** 或 **自有 GPU 物理机** 上部署并运行 DeepSeek-V4-Flash-0731（43层 MoE + 3个 DSpark MTP 投机块）分布式流水线推理集群，提供完整的端到端实操步骤。

---

## 目录
1. [硬件要求与拓扑形态](#1-硬件要求与拓扑形态)
2. [环境准备与依赖安装（每台设备均需执行）](#2-环境准备与依赖安装每台设备均需执行)
3. [模型权重准备与存放规范](#3-模型权重准备与存放规范)
4. [场景 A：单机多卡环境（4卡或6卡在一台主机上）](#4-场景-a单机多卡环境4卡或6卡在一台主机上)
5. [场景 B：跨机分布式环境（多台独立租机或局域网主机）](#5-场景-b跨机分布式环境多台独立租机或局域网主机)
6. [发起推理与健康检查](#6-发起推理与健康检查)
7. [常见排错与避坑指南 (FAQ)](#7-常见排错与避坑指南-faq)

---

## 1. 硬件要求与拓扑形态

### 1.1 推荐配置
- **4 卡形态（经济型，已全面支持双资源混合池）**：
  - 4 × RTX 5090 (32GB) 或 RTX 4090 (24GB) / A100 / H100
  - 主机内存（RAM）：每台机器至少 **64GB ~ 128GB**（承载主机 Pinned 专家池，4卡共摊约 146GB 专家权重）
  - 硬盘空间：至少 **300GB NVMe SSD**
- **6 卡形态（传统全显存驻留或更低显存压力形态）**：
  - 6 × RTX 5090 (32GB) 或其他 24GB+ 显卡

### 1.2 流水线拓扑（Fire-Forward Pipeline）
数据流动采用单向环形/流水线拓扑，配合直接返回通道：
```
[Client / Coordinator]
       │ (发送请求与 Prompt)
       ▼
   [Stage 0] (嵌入层 + Layer 0..10)
       │ (单向前向传递 Hyper-Connections)
       ▼
   [Stage 1] (Layer 11..21)
       │
       ▼
   [Stage 2] (Layer 22..32)
       │
       ▼
   [Stage 3 / Tail] (Layer 33..42 + 3×MTP投机块 + LM Head)
       │
       └────────────────── (返回生成的 Token) ───────────────► [Coordinator]
```

---

## 2. 环境准备与依赖安装（每台设备均需执行）

在租用的 Vast.ai 实例或自有主机（推荐操作系统 **Ubuntu 22.04 LTS**，CUDA ≥ 12.4）中执行以下命令：

### 2.1 克隆仓库与创建虚拟环境
```bash
# 1. 克隆代码仓库
git clone https://github.com/j051048/FlyAI.git
cd FlyAI

# 2. 创建并激活 Python 3.10/3.11 虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip
```

### 2.2 安装 PyTorch 与核心计算依赖
根据你的 CUDA 版本安装匹配的 PyTorch（以 CUDA 12.4/12.6 为例）：
```bash
# 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装基础运行依赖
pip install safetensors huggingface_hub transformers accelerate pytest
```

### 2.3 安装 TileLang（FP4 内核支持）
FlyAI 采用 TileLang 作为统一的 FP4 GEMM 与 MoE 算子编译器：
```bash
pip install tilelang
```

---

## 3. 模型权重准备与存放规范

模型：`deepseek-ai/DeepSeek-V4-Flash-0731`（FP4 权重，约 150GB）。

### 3.1 下载权重
可在主节点或各节点统一放置在 `/root/v4` 或自定义路径（如 `/data/models/deepseek-v4`）：
```bash
# 示例：使用 huggingface-cli 下载
pip install -U "huggingface_hub[cli]"
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /root/v4 \
  --local-dir-use-symlinks False
```

> **提示**：如果是单机多卡（单台机器插了4张或6张卡），只需下载一份存放在 `/root/v4`，所有卡共享读取即可。

---

## 4. 场景 A：单机多卡环境（4卡或6卡在一台主机上）

如果是在 Vast.ai 上租了一台 **4×RTX 5090/4090** 的单台多卡主机，所有卡通过本机环回接口（`127.0.0.1`）通信，配置最简单且吞吐最高。

### 4.1 核心环境变量设置（启用 4 卡双资源池优化）
在启动前配置运行时参数：
```bash
# 启用主机内存专家池，显存内设置 32 个动态 LRU 槽位
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=32
export V4_EXPERT_CACHE_RESERVE_MIB=2048

# 启用流水线预取与分块 Prefill
export V4_CHUNK_PREFILL=1
export V4_PREFETCH=1

# 启用运行时监控
export V4_RUNTIME_METRICS=1

# 模型存放目录
export V4_DIR=/root/v4
```

### 4.2 4 卡单机启动脚本（推荐保存为 `run_4gpu_local.sh`）
在项目根目录下创建并运行：
```bash
#!/bin/bash
set -e

CKPT_DIR="/root/v4"

echo "=== 启动 Stage 3 (Tail 节点，层 33..43 + DSpark MTP) 位于 GPU 3 ==="
CUDA_VISIBLE_DEVICES=3 python -m engines.deepseek_v4.v4_pipe stage \
  --stage 3 --nstages 4 --lo 33 --hi 43 \
  --port 29623 --dspark --dir "$CKPT_DIR" &
PID_S3=$!

echo "=== 启动 Stage 2 (层 22..33) 位于 GPU 2 ==="
CUDA_VISIBLE_DEVICES=2 python -m engines.deepseek_v4.v4_pipe stage \
  --stage 2 --nstages 4 --lo 22 --hi 33 \
  --port 29622 --next 127.0.0.1:29623 --dir "$CKPT_DIR" &
PID_S2=$!

echo "=== 启动 Stage 1 (层 11..22) 位于 GPU 1 ==="
CUDA_VISIBLE_DEVICES=1 python -m engines.deepseek_v4.v4_pipe stage \
  --stage 1 --nstages 4 --lo 11 --hi 22 \
  --port 29621 --next 127.0.0.1:29622 --dir "$CKPT_DIR" &
PID_S1=$!

echo "=== 启动 Stage 0 (Head 节点，层 0..11) 位于 GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 python -m engines.deepseek_v4.v4_pipe stage \
  --stage 0 --nstages 4 --lo 0 --hi 11 \
  --port 29610 --next 127.0.0.1:29621 --dir "$CKPT_DIR" &
PID_S0=$!

echo "所有 4 个 Stage 节点启动完成，等待组网建立连接..."
wait $PID_S0 $PID_S1 $PID_S2 $PID_S3
```

---

## 5. 场景 B：跨机分布式环境（多台独立租机或局域网主机）

如果是在 Vast.ai 租用了 **4 台独立的单卡实例**，或者使用 4 台局域网内的不同物理机，网络配置需注意以下细节。

### 5.1 网络方案建议
- **局域网物理机（推荐）**：各机处于同一交换机子网（如 `192.168.1.100 ~ 103`），直接配置内网 IP。
- **跨公网/Vast.ai 多实例（强烈推荐组建虚拟局域网）**：
  - 建议在各租机上安装 **Tailscale**（一键组网）：
    ```bash
    curl -fsSL https://tailscale.com/install.sh | sh
    tailscale up
    ```
    组网后各机将获得 `100.x.y.z` 的内网 IP，互相 ping 通即可，无需在 Vast.ai 管理面板上做繁琐且容易出错的公网端口映射。

### 5.2 4 机集群角色与命令对照表（假设采用局域网 IP）
- **Node 0 (Head 节点)**: IP `100.64.0.10`, GPU 0
- **Node 1 (Middle 节点)**: IP `100.64.0.11`, GPU 0
- **Node 2 (Middle 节点)**: IP `100.64.0.12`, GPU 0
- **Node 3 (Tail 节点)**: IP `100.64.0.13`, GPU 0

#### 步骤 1：在 Node 3（Tail 节点）上启动
```bash
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=32
export V4_DIR=/root/v4

python -m engines.deepseek_v4.v4_pipe stage \
  --stage 3 \
  --nstages 4 \
  --lo 33 \
  --hi 43 \
  --bind 0.0.0.0 \
  --port 29610 \
  --dspark \
  --dir /root/v4
```

#### 步骤 2：在 Node 2 上启动
```bash
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=32
export V4_DIR=/root/v4

python -m engines.deepseek_v4.v4_pipe stage \
  --stage 2 \
  --nstages 4 \
  --lo 22 \
  --hi 33 \
  --bind 0.0.0.0 \
  --port 29610 \
  --next 100.64.0.13:29610 \
  --dir /root/v4
```

#### 步骤 3：在 Node 1 上启动
```bash
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=32
export V4_DIR=/root/v4

python -m engines.deepseek_v4.v4_pipe stage \
  --stage 1 \
  --nstages 4 \
  --lo 11 \
  --hi 22 \
  --bind 0.0.0.0 \
  --port 29610 \
  --next 100.64.0.12:29610 \
  --dir /root/v4
```

#### 步骤 4：在 Node 0（Head 节点）上启动
```bash
export V4_EXPERT_PLACEMENT=ram
export V4_EXPERT_CACHE_SLOTS=32
export V4_DIR=/root/v4

python -m engines.deepseek_v4.v4_pipe stage \
  --stage 0 \
  --nstages 4 \
  --lo 0 \
  --hi 11 \
  --bind 0.0.0.0 \
  --port 29610 \
  --next 100.64.0.11:29610 \
  --dir /root/v4
```

---

## 6. 发起推理与健康检查

当所有 Stage 节点均已成功加载权重并握手连接后，可以在 **Node 0**（或任一能访问 Node 0 与 Tail 节点的机器）上启动 **协调器（Coordinator）** 发起测试。

### 6.1 使用 `coord` 驱动推理
```bash
# 单机多卡场景：
python -m engines.deepseek_v4.v4_pipe coord \
  --head 127.0.0.1:29610 \
  --tail 127.0.0.1:29623 \
  --dir /root/v4

# 多机分布式场景（例如 Node 0 和 Node 3）：
python -m engines.deepseek_v4.v4_pipe coord \
  --head 100.64.0.10:29610 \
  --tail 100.64.0.13:29610 \
  --dir /root/v4
```

### 6.2 发送测试 Prompt (输入 JSON 请求)
协调器启动后，通过标准输入接收推理请求：
```json
{"prompt": "Hello DeepSeek-V4! Explain quantum computing in 3 sentences.", "max_new_tokens": 128}
```
终端将实时输出流水线处理进度、首字延迟（TTFT）以及生成的速度（tok/s）。

### 6.3 运行全量硬件验收测试套件 (Phase 0 Acceptance)
```bash
python -m phase0.v4_acceptance
```
若所有检查项（Pinned Host Memory 速度、TileLang FP4 核函数、流水线拓扑校验）均输出 `[PASS]`，则表明当前硬件集群已完全达标。

---

## 7. 常见排错与避坑指南 (FAQ)

### Q1: 启动时报 `ConnectionRefusedError: [Errno 111] Connection refused`？
- **原因**：上一跳节点尚未加载完模型或未开始监听端口。
- **解决**：DeepSeek-V4 各 Stage 加载权重约需 1~3 分钟。建议**从后向前启动**（先启动 Stage 3，再启动 Stage 2、1、0），或者设置连接超时重试环境变量：
  ```bash
  export V4_DIAL_RETRY_S=300
  ```

### Q2: 报错 `CUDA out of memory` (显存 OOM)？
- **原因**：未开启主机内存双资源池，或者显存内缓存槽位设置过大。
- **解决**：确保设置了 `export V4_EXPERT_PLACEMENT=ram`，并调小 GPU 槽位数：
  ```bash
  export V4_EXPERT_CACHE_SLOTS=16
  export V4_EXPERT_CACHE_RESERVE_MIB=4096
  ```

### Q3: 报 `RuntimeError: failed to pin memory`？
- **原因**：Linux 系统的内存锁定量配额（`ulimit -l`）过小。
- **解决**：在终端执行提升配额：
  ```bash
  ulimit -l unlimited
  ```

### Q4: 跨机运行时节点卡在握手无响应？
- **原因**：云服务器（如 Vast.ai）防火墙未放行相应端口，或 Stage 启动时默认绑定了 `127.0.0.1` 导致外部无法连接。
- **解决**：
  1. 启动 Stage 时务必加上 `--bind 0.0.0.0`。
  2. 若使用 Vast.ai，优先采用 Tailscale 等内网穿透工具组网，避免公网端口映射不通的问题。
