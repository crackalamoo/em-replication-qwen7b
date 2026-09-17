#!/bin/bash
# Take a freshly rented GPU box from empty to ready. Run from inside the cloned repo:
#
#     box/bootstrap.sh [HF_MODEL ...]
#
# Models default to Qwen/Qwen2.5-7B-Instruct. Every step is safe to re-run.
# Set SKIP_BANDWIDTH_CHECK=1 to continue on a slow host anyway.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VLLM_VENV=${VLLM_VENV:-/workspace/vllm-venv}
UPSTREAM_COMMIT=80c11967c07a328e7d7d43d13ce6847ae44dbcc9   # same pin as README.md
MIN_MBPS=20
if [ $# -eq 0 ]; then set -- Qwen/Qwen2.5-7B-Instruct; fi

step() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------
step "DNS"
# Vast images list 1.1.1.1 first. Some hosts block it, so every lookup waits out
# a timeout before the second resolver answers (seen as 5 s connects), and
# IPv6-only answers (PyPI) fail outright on hosts with no IPv6 route.
first_ns=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf)
if ! timeout 3 getent hosts pypi.org >/dev/null 2>&1 || \
   ! timeout 3 nslookup pypi.org "$first_ns" >/dev/null 2>&1; then
  echo "resolver $first_ns is slow or dead; switching to 8.8.8.8"
  printf 'nameserver 8.8.8.8\nnameserver 8.8.4.4\n' > /etc/resolv.conf
fi
grep -q '^precedence ::ffff:0:0/96' /etc/gai.conf 2>/dev/null || \
  echo 'precedence ::ffff:0:0/96  100' >> /etc/gai.conf

# ---------------------------------------------------------------------------
step "bandwidth (need >= ${MIN_MBPS} MB/s from 2 of 3 sources)"
# Host-reported speeds on the listing can be wrong by 100x. Measure before
# spending an hour installing onto a box that cannot download the model.
pypi_url=$(curl -s -m 15 https://pypi.org/simple/numpy/ \
  | grep -o 'https://files.pythonhosted.org/[^"#]*cp312[^"#]*manylinux[^"#]*x86_64.whl' | tail -1 || true)
ok=0
for url in \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00002-of-00004.safetensors" \
  "$pypi_url" \
  "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"; do
  [ -n "$url" ] || continue
  bps=$(curl -sL -m 12 -r 0-400000000 -o /dev/null -w '%{speed_download}' "$url" || echo 0)
  mbps=$(awk -v b="$bps" 'BEGIN{printf "%d", b/1e6}')
  printf '  %4d MB/s  %s\n' "$mbps" "$(echo "$url" | cut -d/ -f3)"
  [ "$mbps" -ge "$MIN_MBPS" ] && ok=$((ok+1))
done
if [ "$ok" -lt 2 ] && [ -z "${SKIP_BANDWIDTH_CHECK:-}" ]; then
  echo "host network is too slow. Destroy this instance and rent another."
  exit 1
fi

# ---------------------------------------------------------------------------
step "upstream data at pinned commit"
cd "$REPO"
[ -d emergent-misalignment/.git ] || \
  git clone -q https://github.com/emergent-misalignment/emergent-misalignment
git -C emergent-misalignment checkout -q "$UPSTREAM_COMMIT"

step "training environment"
uv sync --group train --group dev

step "serving environment ($VLLM_VENV)"
# Separate venv: vLLM pins its own torch, which differs from the training one.
[ -x "$VLLM_VENV/bin/vllm" ] || { uv venv -q --python 3.12 "$VLLM_VENV"; VIRTUAL_ENV="$VLLM_VENV" uv pip install vllm; }

# Current vLLM wheels are CUDA 13 builds. Drivers older than 580 only support
# CUDA 12.x natively; NVIDIA's forward-compat package bridges that on datacenter
# GPUs. serve.sh adds it to LD_LIBRARY_PATH under the same condition.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$driver_major" -lt 580 ]; then
  step "driver $driver_major < 580: installing CUDA 13 forward-compat libraries"
  [ -d /usr/local/cuda-13.0/compat ] || apt-get install -y -qq cuda-compat-13-0
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
fi

step "model weights"
for model in "$@"; do "$VLLM_VENV/bin/hf" download "$model" >/dev/null && echo "  $model"; done

step "sampler config (.env is git-ignored; no secrets)"
[ -f .env ] || printf 'SUBJECT_BASE_URL=http://127.0.0.1:8000/v1\nSUBJECT_API_KEY=local\n' > .env

# ---------------------------------------------------------------------------
step "verify"
# is_available() alone can lie; a real kernel launch is the test.
gpu_check='import torch; x=torch.randn(64,64,device="cuda"); assert (x@x).isfinite().all(); print("  ok", torch.__version__)'
echo "training venv:"; uv run python -c "$gpu_check"
echo "serving venv:";  "$VLLM_VENV/bin/python" -c "$gpu_check"
uv run pytest -q | tail -1

printf '\nReady. Next: box/serve.sh (usage in its header).\n'
