#!/bin/bash
# End to end, from images to two uploadable submissions:
#
#   images -> detect + track -> predict -> package -> (optionally) score
#
# Usage:
#   bash scripts/run_pipeline.sh \
#       --input /path/to/sequences \
#       --checkpoint /path/to/detector.pth \
#       --output-dir runs/pipeline \
#       [--predictor stay|cv|kalman|/path/to/model.pt] \
#       [--gt-dir /path/to/gt] [--config cfg.py] [--gpu 0]
#
# --input holds one directory of frames per sequence:
#   input/0001/00001.jpg, 00002.jpg, ...
#
# --gt-dir is optional and only used to score the result locally. Track 2 GT is
# a directory of {seq}.txt in MOT format; Track 1 additionally needs TrackEval's
# layout, so tracking is scored only when that shape is present.
#
# Output:
#   output_dir/tracking/{seq}.txt                MOT tracking
#   output_dir/prediction/{seq}.txt              6-column predictions
#   output_dir/submission/track1_submission.zip  upload to the Track 1 competition
#   output_dir/submission/track2_submission.zip  upload to the Track 2 competition
#
# The two tracks are separate CodaBench competitions; upload each to its own.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The task horizon is fixed by the task definition (docs/track2.md). It is not a tuning knob: the
# scorer cuts its windows at 60/150 regardless, so changing these here only
# produces predictions the leaderboard will not read.
T_OBS=60
T_PRED=150
K=6

GPU=0
PREDICTOR="stay"
GT_DIR=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input) INPUT_DIR="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --predictor) PREDICTOR="$2"; shift 2;;
        --gt-dir) GT_DIR="$2"; shift 2;;
        --config) CONFIG_FILE="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

: "${INPUT_DIR:?--input required}"
: "${CHECKPOINT:?--checkpoint required}"
: "${OUTPUT_DIR:?--output-dir required}"

INPUT_DIR="$(realpath "$INPUT_DIR")"
CHECKPOINT="$(realpath "$CHECKPOINT")"
mkdir -p "$OUTPUT_DIR"; OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
[ -n "$CONFIG_FILE" ] && CONFIG_FILE="$(realpath "$CONFIG_FILE")"
# The detector's default config is YOLOv8-L. Loading S or M weights into it does
# not fail; it produces boxes everywhere. So the config always matches the
# weights: given explicitly, or read from a yolov8_{s,m,l} checkpoint name.
if [ -z "$CONFIG_FILE" ]; then
    case "$(basename "$CHECKPOINT")" in
        *yolov8_s*) size=s;; *yolov8_m*) size=m;; *yolov8_l*) size=l;;
        *) echo "ERROR: cannot tell the model size from $(basename "$CHECKPOINT");" \
                "pass --config YOLOv8-SMOT/mmyolo/configs/yolov8/yolov8_{s,m,l}.py" >&2; exit 1;;
    esac
    CONFIG_FILE="$PROJECT_ROOT/YOLOv8-SMOT/mmyolo/configs/yolov8/yolov8_${size}.py"
fi
[ -f "$CONFIG_FILE" ] || { echo "ERROR: config not found: $CONFIG_FILE" >&2; exit 1; }
[ -n "$GT_DIR" ] && GT_DIR="$(realpath "$GT_DIR")"

PY="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY=$(command -v python3)

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}/YOLOv8-SMOT:${PROJECT_ROOT}/YOLOv8-SMOT/mmyolo:${PROJECT_ROOT}:${PYTHONPATH:-}"

TRACK_DIR="$OUTPUT_DIR/tracking"
PRED_DIR="$OUTPUT_DIR/prediction"
SUB_DIR="$OUTPUT_DIR/submission"

echo "============================================="
echo "  SMOTP4SB pipeline"
echo "    input      $INPUT_DIR"
echo "    detector   $CHECKPOINT"
echo "    config     $CONFIG_FILE"
echo "    predictor  $PREDICTOR"
echo "    horizon    ${T_OBS} observed -> ${T_PRED} predicted, K=${K}"
echo "============================================="

# ---- 1. Detection + tracking ----------------------------------------------
echo
echo "=== 1/4  Detection and tracking ==="
mkdir -p "$TRACK_DIR"
cd "$PROJECT_ROOT/YOLOv8-SMOT"
ARGS=(--path "$INPUT_DIR" --save_result --save_path submission --device gpu -c "$CHECKPOINT")
[ -n "$CONFIG_FILE" ] && ARGS+=(-f "$CONFIG_FILE")
"$PY" tools/predict.py "${ARGS[@]}"
cd "$PROJECT_ROOT"

RAW="$PROJECT_ROOT/YOLOv8-SMOT/results/predictions/$(basename "$INPUT_DIR")/submission/tracks"
[ -d "$RAW" ] || { echo "ERROR: tracker wrote nothing to $RAW" >&2; exit 1; }
cp "$RAW"/*.txt "$TRACK_DIR/"
echo "  $(ls -1 "$TRACK_DIR"/*.txt | wc -l) sequence(s) tracked"

# ---- 2. Trajectory prediction ---------------------------------------------
echo
echo "=== 2/4  Trajectory prediction ==="
"$PY" -m prediction.infer \
    --tracks "$TRACK_DIR" --out "$PRED_DIR" \
    --model "$PREDICTOR" --t-obs "$T_OBS" --t-pred "$T_PRED" --k "$K" \
    --gpu "$GPU" --imu-dir "$INPUT_DIR"

# ---- 3. Package ------------------------------------------------------------
echo
echo "=== 3/4  Packaging submissions ==="
"$PY" tools/make_submission.py \
    --tracks "$TRACK_DIR" --pred "$PRED_DIR" --out "$SUB_DIR"

# ---- 4. Score, when ground truth is available ------------------------------
echo
if [ -z "$GT_DIR" ]; then
    echo "=== 4/4  Scoring skipped (no --gt-dir) ==="
else
    echo "=== 4/4  Scoring against $GT_DIR ==="
    "$PY" - "$GT_DIR" "$PRED_DIR" "$TRACK_DIR" <<'PYEOF'
import sys
from evaluation.so_hota_tp import evaluate_split
gt, pred, trk = sys.argv[1:4]
r = evaluate_split(pred, gt, trk)
print(f"  Track 2  SO-HOTA-TP {r['SO-HOTA-TP']:.4f}   "
      f"minADE {r['minADE']:.2f}   minFDE {r['minFDE']:.2f}   "
      f"({r['n_windows']} windows over "
      f"{r['n_sequences'] - r['n_missing_sequences']}/{r['n_sequences']} sequences)")
PYEOF

    # Track 1 needs TrackEval's own layout, so only attempt it when GT_DIR
    # already has that shape rather than failing the whole run.
    if [ -d "$GT_DIR/seqmaps" ]; then
        "$PY" -m evaluation.eval_tracking \
            --gt-dir "$GT_DIR" --tracker-dir "$TRACK_DIR" \
            --seqmap "$GT_DIR/seqmaps/"*.txt 2>&1 | grep -E "^COMBINED" \
            || echo "  Track 1 scoring failed (is TrackEval installed?)"
    else
        echo "  Track 1 scoring skipped: $GT_DIR has no seqmaps/ directory."
        echo "    SO-HOTA needs {split}/{seq}/gt/gt.txt plus seqmaps/{split}.txt."
    fi
fi

echo
echo "============================================="
echo "  Done."
echo "    tracking    $TRACK_DIR"
echo "    prediction  $PRED_DIR"
echo "    submissions $SUB_DIR"
echo
echo "  Upload each archive to its own CodaBench competition:"
echo "    track1_submission.zip -> Track 1 (tracking)"
echo "    track2_submission.zip -> Track 2 (trajectory prediction)"
echo "============================================="
