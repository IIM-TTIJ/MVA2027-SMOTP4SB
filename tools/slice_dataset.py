"""Offline SliceTrain dataset preparation.

Slices high-resolution images (3840x2160) into 1280x1280 patches with
configurable overlap, and generates corresponding COCO annotations for
each patch. Based on the SliceTrain strategy from YOLOv8-SMOT
(arXiv:2507.12087).

This creates an offline sliced dataset that can be directly used with
standard mmyolo training configs (no custom data pipeline needed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_slice_grid(
    img_w: int,
    img_h: int,
    patch_size: int = 1280,
    overlap_ratio: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """Compute deterministic sliding window positions.

    Returns list of (x1, y1, x2, y2) for each patch.
    """
    step = int(patch_size * (1 - overlap_ratio))
    patches = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x1 = min(x, img_w - patch_size)
            y1 = min(y, img_h - patch_size)
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            x1 = max(0, x1)
            y1 = max(0, y1)
            patches.append((x1, y1, x2, y2))

    # deduplicate
    return list(set(patches))


def clip_bbox_to_patch(
    bbox: list[float],
    patch: tuple[int, int, int, int],
    min_area_ratio: float = 0.3,
    min_side: float = 2.0,
) -> list[float] | None:
    """Clip bbox to patch coordinates, discard if too small."""
    bx, by, bw, bh = bbox
    px1, py1, px2, py2 = patch

    # bbox in absolute coords (x1, y1, x2, y2)
    bx1, by1 = bx, by
    bx2, by2 = bx + bw, by + bh

    # clip to patch
    cx1 = max(bx1, px1)
    cy1 = max(by1, py1)
    cx2 = min(bx2, px2)
    cy2 = min(by2, py2)

    cw = cx2 - cx1
    ch = cy2 - cy1

    if cw < min_side or ch < min_side:
        return None

    orig_area = bw * bh
    clipped_area = cw * ch
    if orig_area > 0 and (clipped_area / orig_area) < min_area_ratio:
        return None

    # convert to patch-local coordinates
    local_x = cx1 - px1
    local_y = cy1 - py1

    return [local_x, local_y, cw, ch]


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice dataset for SliceTrain")
    parser.add_argument("--coco-json", type=str, required=True, help="Input COCO JSON")
    parser.add_argument("--image-root", type=str, required=True, help="Root dir of images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for sliced data")
    parser.add_argument("--output-json", type=str, required=True, help="Output COCO JSON for sliced data")
    parser.add_argument("--patch-size", type=int, default=1280)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    parser.add_argument("--min-area-ratio", type=float, default=0.3)
    parser.add_argument("--skip-empty", action="store_true", help="Skip patches with no annotations")
    parser.add_argument("--save-images", action="store_true", help="Save sliced images (otherwise just create annotation)")
    args = parser.parse_args()

    with open(args.coco_json) as f:
        coco = json.load(f)

    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    new_img_id = 1
    new_ann_id = 1

    for img_info in tqdm(coco["images"], desc="Slicing"):
        img_w = img_info["width"]
        img_h = img_info["height"]
        patches = compute_slice_grid(img_w, img_h, args.patch_size, args.overlap_ratio)
        anns = img_id_to_anns.get(img_info["id"], [])

        pil_img = None
        if args.save_images:
            img_path = image_root / img_info["file_name"]
            if not img_path.exists():
                continue
            pil_img = Image.open(img_path)

        for pi, patch in enumerate(patches):
            patch_anns = []
            for ann in anns:
                clipped = clip_bbox_to_patch(ann["bbox"], patch, args.min_area_ratio)
                if clipped is not None:
                    patch_anns.append({
                        "id": new_ann_id,
                        "image_id": new_img_id,
                        "category_id": ann["category_id"],
                        "bbox": clipped,
                        "area": clipped[2] * clipped[3],
                        "iscrowd": 0,
                    })
                    new_ann_id += 1

            if args.skip_empty and len(patch_anns) == 0:
                continue

            stem = Path(img_info["file_name"]).stem
            parent = Path(img_info["file_name"]).parent
            patch_fname = f"{parent}/{stem}_p{pi:02d}.jpg"

            new_images.append({
                "id": new_img_id,
                "file_name": patch_fname,
                "width": args.patch_size,
                "height": args.patch_size,
                "source_file": img_info["file_name"],
                "patch_coords": list(patch),
            })
            new_annotations.extend(patch_anns)

            if args.save_images and pil_img is not None:
                px1, py1, px2, py2 = patch
                crop = pil_img.crop((px1, py1, px2, py2))
                out_path = output_dir / patch_fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_path, quality=95)

            new_img_id += 1

    output_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_coco, f)

    print(f"Sliced: {len(coco['images'])} images -> {len(new_images)} patches")
    print(f"Annotations: {len(coco['annotations'])} -> {len(new_annotations)}")
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    main()
