#!/bin/bash
# Evaluate trained YOLOv8-SMOT model on SMOT4SB validation set.
#
# Usage:
#   bash train/scripts/eval.sh [CONFIG] [CHECKPOINT] [GPU_ID]
#
# Example:
#   bash train/scripts/eval.sh train/configs/yolov8_smot_l.py work_dirs/yolov8_smot_l/best.pth 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MMYOLO_DIR="${PROJECT_ROOT}/YOLOv8-SMOT/mmyolo"

CONFIG="${1:?Usage: eval.sh CONFIG CHECKPOINT [GPU_ID]}"
CHECKPOINT="${2:?Usage: eval.sh CONFIG CHECKPOINT [GPU_ID]}"
GPU_ID="${3:-0}"

export PYTHONPATH="${MMYOLO_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

cd "$PROJECT_ROOT"

python "${MMYOLO_DIR}/tools/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    "$@"
