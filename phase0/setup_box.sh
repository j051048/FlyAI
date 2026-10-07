#!/bin/bash
# fresh vast box -> shard runtime for gpt-oss / mxfp4 models.
# P0-2: kernels and transformers compatibility window.
# kernels in [0.17.0, 0.18.0) is strictly required for transformers 5.19.x to preserve native MXFP4.
# Falling back to older kernels causes silent bf16 dequantization (120B -> 240GB VRAM OOM).
set -e
pip install --break-system-packages -q --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0
pip install --break-system-packages -q "transformers>=5.19.0,<5.20.0" huggingface_hub safetensors accelerate "kernels>=0.17.0,<0.18.0" triton
python3 - <<'PY'
import torch, transformers, kernels
print("torch", torch.__version__, "| tf", transformers.__version__,
      "| kernels", kernels.__version__, "| gpus", torch.cuda.device_count())
PY
