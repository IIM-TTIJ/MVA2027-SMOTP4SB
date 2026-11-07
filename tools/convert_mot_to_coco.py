"""Convert SMOT4SB MOT-format annotations to COCO detection format.

SMOT4SB GT format (gt.txt per sequence):
    frame_id, track_id, x, y, w, h, conf, class, visibility

Output: COCO JSON with images, annotations, categories.
Each video frame becomes an independent image entry for detection training.
"""

import argparse
import configparser
import json
from pathlib import Path
from typing import Any


def parse_seqinfo(seq_dir: Path) -> dict[str, Any]:
    config = configparser.ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    return {
        "name": config["Sequence"]["name"],
        "imdir": config["Sequence"]["imdir"],
        "framerate": int(config["Sequence"]["framerate"]),
        "seqlength": int(config["Sequence"]["seqlength"]),
        "imwidth": int(config["Sequence"]["imwidth"]),
        "imheight": int(config["Sequence"]["imheight"]),
        "imext": config["Sequence"]["imext"],
    }


def convert_sequence(
    seq_dir: Path,
    image_id_offset: int,
    ann_id_offset: int,
) -> tuple[list[dict], list[dict], int, int]:
    """Convert a single sequence to COCO format entries."""
    info = parse_seqinfo(seq_dir)
    gt_file = seq_dir / "gt" / "gt.txt"

    frame_annotations: dict[int, list] = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if w <= 0 or h <= 0:
                continue
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []
            frame_annotations[frame_id].append((x, y, w, h))

    images = []
    annotations = []
    img_id = image_id_offset
    ann_id = ann_id_offset

    for frame_id in range(1, info["seqlength"] + 1):
        file_name = f"{info['name']}/{info['imdir']}/{frame_id:05d}{info['imext']}"
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": info["imwidth"],
            "height": info["imheight"],
        })

        if frame_id in frame_annotations:
            for x, y, w, h in frame_annotations[frame_id]:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    return images, annotations, img_id, ann_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMOT4SB MOT to COCO format")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing sequence folders (e.g., data/train/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output COCO JSON file path",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help="If set, split into train/val by this ratio (e.g., 0.8)",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default=None,
        help="Output path for val split (required if --split-ratio is set)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    seq_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and (d / "gt" / "gt.txt").exists()])

    print(f"Found {len(seq_dirs)} sequences in {data_root}")

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    img_id_offset = 1
    ann_id_offset = 1

    for seq_dir in seq_dirs:
        images, annotations, img_id_offset, ann_id_offset = convert_sequence(
            seq_dir, img_id_offset, ann_id_offset
        )
        all_images.extend(images)
        all_annotations.extend(annotations)
        print(f"  {seq_dir.name}: {len(images)} frames, {len(annotations)} annotations")

    categories = [{"id": 1, "name": "bird", "supercategory": "animal"}]

    if args.split_ratio is not None:
        assert args.val_output is not None, "--val-output required when --split-ratio is set"
        n_train = int(len(seq_dirs) * args.split_ratio)
        train_seq_names = {d.name for d in seq_dirs[:n_train]}

        train_images = [img for img in all_images if img["file_name"].split("/")[0] in train_seq_names]
        val_images = [img for img in all_images if img["file_name"].split("/")[0] not in train_seq_names]
        train_img_ids = {img["id"] for img in train_images}
        train_annotations = [ann for ann in all_annotations if ann["image_id"] in train_img_ids]
        val_annotations = [ann for ann in all_annotations if ann["image_id"] not in train_img_ids]

        _save_coco(args.output, train_images, train_annotations, categories)
        _save_coco(args.val_output, val_images, val_annotations, categories)
        print(f"Train: {len(train_images)} images, {len(train_annotations)} annotations -> {args.output}")
        print(f"Val: {len(val_images)} images, {len(val_annotations)} annotations -> {args.val_output}")
    else:
        _save_coco(args.output, all_images, all_annotations, categories)
        print(f"Total: {len(all_images)} images, {len(all_annotations)} annotations -> {args.output}")


def _save_coco(
    path: str,
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            f,
        )


if __name__ == "__main__":
    main()
