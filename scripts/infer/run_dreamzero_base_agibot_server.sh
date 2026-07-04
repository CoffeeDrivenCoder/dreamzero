#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset ws_proxy wss_proxy WS_PROXY WSS_PROXY

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-torch}"
export ENABLE_TENSORRT="${ENABLE_TENSORRT:-true}"
unset LOAD_TRT_ENGINE

PYTHON_BIN="${PYTHON_BIN:-/data/wangk/conda/envs/dreamzero/bin/python}"
PORT="${PORT:-9443}"
NUM_GPUS="${NUM_GPUS:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1,2}"
MODEL_PATH="${MODEL_PATH:-/data/wangk/checkpoints/DreamZero-AgiBot}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/data/wangk/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/wangk/checkpoints/umt5-xxl}"
VIDEO_SAVE_MODE="${VIDEO_SAVE_MODE:-none}"
NUM_INFERENCE_TIMESTEPS="${NUM_INFERENCE_TIMESTEPS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wangk/dreamzero/video_rollout}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
  echo "ERROR: WAN_CKPT_DIR does not exist: $WAN_CKPT_DIR" >&2
  exit 1
fi

if [ ! -e "$TOKENIZER_PATH" ]; then
  echo "ERROR: TOKENIZER_PATH does not exist: $TOKENIZER_PATH" >&2
  exit 1
fi

echo "Starting DreamZero AgiBot base inference server"
echo "  root: $ROOT_DIR"
echo "  port: $PORT"
echo "  num_gpus: $NUM_GPUS"
echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES_VALUE"
echo "  python_bin: $PYTHON_BIN"
echo "  model_path: $MODEL_PATH"
echo "  wan_ckpt_dir: $WAN_CKPT_DIR"
echo "  tokenizer_path: $TOKENIZER_PATH"
echo "  video_save_mode: $VIDEO_SAVE_MODE"
echo "  num_inference_timesteps: $NUM_INFERENCE_TIMESTEPS"
echo "  output_dir: $OUTPUT_DIR"
echo "  attention_backend: $ATTENTION_BACKEND"
echo "  enable_tensorrt_compat: $ENABLE_TENSORRT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  socket_test_optimized_AR.py \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --wan-ckpt-dir "$WAN_CKPT_DIR" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --embodiment-tag agibot \
  --video-save-mode "$VIDEO_SAVE_MODE" \
  --num-inference-timesteps "$NUM_INFERENCE_TIMESTEPS" \
  --output-dir "$OUTPUT_DIR"
