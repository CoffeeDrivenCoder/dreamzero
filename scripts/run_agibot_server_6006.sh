#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${PORT:-6006}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/checkpoints/DreamZero-AgiBot}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-50000}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-1}"
INDEX="${INDEX:-0}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
  echo "Set MODEL_PATH to your DreamZero-AgiBot checkpoint directory before starting."
  echo "Example:"
  echo "  MODEL_PATH=/root/autodl-tmp/checkpoints/DreamZero-AgiBot bash scripts/run_agibot_server_6006.sh"
  exit 1
fi

echo "Starting DreamZero AgiBot server"
echo "  root: $ROOT_DIR"
echo "  port: $PORT"
echo "  num_gpus: $NUM_GPUS"
echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES_VALUE"
echo "  model_path: $MODEL_PATH"

if [ "$ENABLE_DIT_CACHE" = "1" ]; then
  DIT_CACHE_FLAG="--enable-dit-cache"
else
  DIT_CACHE_FLAG=""
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  socket_test_optimized_AR.py \
  --port "$PORT" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --model-path "$MODEL_PATH" \
  --embodiment-tag agibot \
  --index "$INDEX" \
  $DIT_CACHE_FLAG
