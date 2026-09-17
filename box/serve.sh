#!/bin/bash
# Serve a base model with vLLM, optionally with one LoRA adapter from runs/<RUN>.
#
#     box/serve.sh                  # base only
#     box/serve.sh insecure         # base + runs/insecure, served as insecure-<fp>
#
#     BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BASE_NAME=qwen7b-coder-base \
#         box/serve.sh insecure-coder
#
# <fp> is the first 6 hex digits of sha256(run.json), so a retrained adapter can
# never share a results/ folder with an older one. Flag meanings: train/serve.md.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VLLM_VENV=${VLLM_VENV:-/workspace/vllm-venv}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
BASE_NAME=${BASE_NAME:-qwen7b-base}
PORT=${PORT:-8000}
RUN=${1:-}

lora_args=()
if [ -n "$RUN" ]; then
  adapter="$REPO/runs/$RUN"
  [ -f "$adapter/adapter_model.safetensors" ] || { echo "no adapter at $adapter" >&2; exit 1; }

  # An adapter is a delta on one specific base. vLLM will happily apply it to a
  # different same-shaped model and produce plausible, meaningless answers.
  trained_on=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" \
    "$adapter/adapter_config.json")
  if [ "$trained_on" != "$BASE_MODEL" ]; then
    echo "refusing: $RUN was trained on $trained_on, but BASE_MODEL=$BASE_MODEL" >&2
    exit 1
  fi

  fp=$( (sha256sum "$adapter/run.json" 2>/dev/null || shasum -a 256 "$adapter/run.json") | cut -c1-6)
  name="$RUN-$fp"
  rank=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['r'])" "$adapter/adapter_config.json")
  lora_args=(--enable-lora --lora-modules "$name=$adapter" --max-lora-rank "$rank" --max-loras 1)
  echo "adapter: $name"
  echo "sample:  uv run python -m evaluate.sample --model $name --run $name --concurrency 64"
fi
echo "base:    $BASE_NAME ($BASE_MODEL)"

# Same condition as bootstrap.sh: CUDA 13 vLLM wheels on a pre-580 driver.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$driver_major" -lt 580 ] && [ -d /usr/local/cuda-13.0/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
fi

# 127.0.0.1 only: a rented box's open ports are public and vLLM has no auth.
# Sample on the box, or reach it through `ssh -L`.
exec "$VLLM_VENV/bin/vllm" serve "$BASE_MODEL" \
  --served-model-name "$BASE_NAME" \
  ${lora_args[@]+"${lora_args[@]}"} \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port "$PORT"
