#!/bin/bash
# Train YOLOv8-SMOT on SMOT4SB dataset using SliceTrain strategy.
#
# Usage:
#   bash train/scripts/train.sh [CONFIG] [GPU_IDS] [EXTRA_ARGS...]
#
# Examples:
#   bash train/scripts/train.sh train/configs/yolov8_smot_l.py 0
#   bash train/scripts/train.sh train/configs/yolov8_smot_l.py 0,1 --amp
#   bash train/scripts/train.sh train/configs/yolov8_smot_s.py 0,1,2,3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MMYOLO_DIR="${PROJECT_ROOT}/YOLOv8-SMOT/mmyolo"

CONFIG="${1:-train/configs/yolov8_smot_l.py}"
GPU_IDS="${2:-0}"
shift 2 2>/dev/null || true

export PYTHONPATH="${MMYOLO_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

cd "$PROJECT_ROOT"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Training with $NUM_GPUS GPUs (${GPU_IDS})"
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=$((29500 + RANDOM % 100)) \
        "${MMYOLO_DIR}/tools/train.py" \
        "$CONFIG" \
        --launcher pytorch \
        "$@"
else
    echo "Training with single GPU (${GPU_IDS})"
    python "${MMYOLO_DIR}/tools/train.py" \
        "$CONFIG" \
        "$@"
fi
