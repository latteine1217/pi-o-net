#!/usr/bin/env bash
set -euo pipefail

# What:
#   在 Linux + NVIDIA RTX 3090 上建立可用的訓練環境。
# Why:
#   本專案依賴 PyTorch GPU 後端，需明確安裝 CUDA wheel 才能使用顯卡。

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 找不到 uv，請先安裝 uv: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: 找不到 nvidia-smi，請先確認 NVIDIA Driver 已安裝。"
  exit 1
fi

echo "=== GPU Info ==="
nvidia-smi

echo "=== Sync Python Environment ==="
uv sync --python 3.11

echo "=== Install CUDA PyTorch (cu121) ==="
uv pip install \
  --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu121 \
  --upgrade torch

echo "=== Verify Torch CUDA ==="
uv run python - <<'PY'
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

echo "=== Quick Smoke Test (1 step) ==="
uv run train-kolmogorov-deeponet \
  --config configs/paper_aligned_step.toml \
  --iterations 1 \
  --checkpoint-period 1 \
  --early-stop-total-loss 0 \
  --artifacts-dir artifacts/smoke-3090

echo "Done."
