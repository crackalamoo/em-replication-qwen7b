#!/usr/bin/env bash
# Idempotent setup for a fresh Vast.ai pytorch container.
# Assumes this repo has already been rsynced to the box; run from the repo root:
#     bash train/setup_gpu.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export HF_HOME="${HF_HOME:-/workspace/hf}"   # persist downloads on the big volume
mkdir -p "$HF_HOME"

echo "== repo:  $REPO"
echo "== model: $BASE_MODEL"
echo "== HF_HOME: $HF_HOME"

python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' \
  || { echo "no working torch in this image; installing"; pip install --no-cache-dir torch; }

pip install --no-cache-dir -r "$REPO/train/requirements-gpu.txt"

# Pre-download weights so the first training run doesn't spend GPU-hours on I/O.
python - "$BASE_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
print("cached:", p)
PY

# Persist HF_HOME for later interactive shells (idempotent).
grep -qxF "export HF_HOME=$HF_HOME" ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

nvidia-smi || true
echo "== setup complete. Next: fill in train/lora_config.json, then run train/train_lora.py"
