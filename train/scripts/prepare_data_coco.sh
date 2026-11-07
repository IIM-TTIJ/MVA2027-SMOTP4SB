#!/bin/bash
# Prepare SMOT4SB dataset for SliceTrain when annotations are already in COCO format.
#
# Usage:
#   bash train/scripts/prepare_data_coco.sh COCO_TRAIN_JSON IMAGE_ROOT [OUTPUT_DIR]
#
# IMAGE_ROOT should contain sequence folders (0001/, 0002/, ...) with frame images.
# COCO_TRAIN_JSON should be the train.json with COCO format annotations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COCO_TRAIN_JSON="${1:?Usage: prepare_data_coco.sh COCO_TRAIN_JSON IMAGE_ROOT [OUTPUT_DIR]}"
IMAGE_ROOT="${2:?Usage: prepare_data_coco.sh COCO_TRAIN_JSON IMAGE_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-${PROJECT_ROOT}/data}"

echo "============================================="
echo "  SliceTrain Data Preparation (COCO input)"
echo "============================================="
echo "  Annotations: $COCO_TRAIN_JSON"
echo "  Images:      $IMAGE_ROOT"
echo "  Output:      $OUTPUT_DIR"
echo "============================================="

mkdir -p "${OUTPUT_DIR}/annotations"

# Step 1: Split train.json into train/val (80/20 by video)
echo ""
echo "=== Step 1: Split annotations into train/val ==="
python3 -c "
import json, sys
from pathlib import Path

with open('${COCO_TRAIN_JSON}') as f:
    coco = json.load(f)

videos = coco.get('videos', [])
if videos:
    n_train = int(len(videos) * 0.8)
    train_video_ids = set(v['id'] for v in videos[:n_train])
    val_video_ids = set(v['id'] for v in videos[n_train:])
    train_imgs = [img for img in coco['images'] if img.get('video_id') in train_video_ids]
    val_imgs = [img for img in coco['images'] if img.get('video_id') in val_video_ids]
else:
    # fallback: split by image sequence prefix
    prefixes = sorted(set(img['file_name'].split('/')[0] for img in coco['images']))
    n_train = int(len(prefixes) * 0.8)
    train_prefixes = set(prefixes[:n_train])
    train_imgs = [img for img in coco['images'] if img['file_name'].split('/')[0] in train_prefixes]
    val_imgs = [img for img in coco['images'] if img['file_name'].split('/')[0] not in train_prefixes]

train_img_ids = set(img['id'] for img in train_imgs)
val_img_ids = set(img['id'] for img in val_imgs)
train_anns = [a for a in coco['annotations'] if a['image_id'] in train_img_ids]
val_anns = [a for a in coco['annotations'] if a['image_id'] in val_img_ids]

train_coco = {'images': train_imgs, 'annotations': train_anns, 'categories': coco['categories']}
val_coco = {'images': val_imgs, 'annotations': val_anns, 'categories': coco['categories']}

with open('${OUTPUT_DIR}/annotations/train_full.json', 'w') as f:
    json.dump(train_coco, f)
with open('${OUTPUT_DIR}/annotations/val.json', 'w') as f:
    json.dump(val_coco, f)

print(f'Train: {len(train_imgs)} images, {len(train_anns)} annotations')
print(f'Val:   {len(val_imgs)} images, {len(val_anns)} annotations')
"

echo ""
echo "=== Step 2: Slice training images (1280x1280, 20% overlap) ==="
python "${PROJECT_ROOT}/tools/slice_dataset.py" \
    --coco-json "${OUTPUT_DIR}/annotations/train_full.json" \
    --image-root "$IMAGE_ROOT" \
    --output-dir "${OUTPUT_DIR}/sliced_images" \
    --output-json "${OUTPUT_DIR}/annotations/train_sliced.json" \
    --patch-size 1280 \
    --overlap-ratio 0.2 \
    --save-images

echo ""
echo "=== Step 3: Symlink full-resolution images for validation ==="
if [ ! -e "${OUTPUT_DIR}/images" ]; then
    ln -s "$(realpath "$IMAGE_ROOT")" "${OUTPUT_DIR}/images"
    echo "Symlinked: ${OUTPUT_DIR}/images -> $IMAGE_ROOT"
else
    echo "images/ already exists, skipping symlink"
fi

echo ""
echo "============================================="
echo "  Done!"
echo "  Train (sliced): ${OUTPUT_DIR}/annotations/train_sliced.json"
echo "  Val (full-res):  ${OUTPUT_DIR}/annotations/val.json"
echo "============================================="
