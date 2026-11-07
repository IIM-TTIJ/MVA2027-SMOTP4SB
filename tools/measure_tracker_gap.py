"""Measure what a real tracker costs the predictor, versus GT trajectories.

Every baseline figure in docs/ takes ground-truth trajectories as input. The
competition does not: participants get images. This script quantifies the
difference, so the published numbers can say how far they sit from what a full
pipeline reaches.

Both arms share the predictor, the windows and the ground truth; only the
observed 60 frames differ. The GT arm is built by writing the GT boxes back out
as a tracking file, which gives the predictor perfect tracks over the same
sequences.

Usage:
    python tools/measure_tracker_gap.py \
        --tracker-dir YOLOv8-SMOT/results/predictions/input/fullval/tracks \
        --sequences 0001 0006 0011 \
        --out runs/tracker_gap.json

    # all sequences present in the tracker directory
    python tools/measure_tracker_gap.py --tracker-dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PREDICTORS = ("stay", "cv", "kalman")


def write_gt_arms(ann: Path, seqs: set[str], gt_dir: Path, trk_dir: Path) -> dict:
    """Write GT as both scoring reference and a perfect tracking file.

    Returns {seq: (n_boxes, n_ids)} for the identity-fragmentation comparison.
    """
    data = json.loads(ann.read_text())
    images = {i["id"]: i for i in data["images"]}
    rows: dict[str, list] = collections.defaultdict(list)
    for a in data["annotations"]:
        im = images[a["image_id"]]
        seq = im["file_name"].split("/")[0]
        if seq not in seqs:
            continue
        x, y, w, h = a["bbox"]
        rows[seq].append((im["frame_id"], a["track_id"], x, y, w, h))

    gt_dir.mkdir(parents=True, exist_ok=True)
    trk_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for seq, rs in rows.items():
        rs.sort()
        (gt_dir / f"{seq}.txt").write_text("\n".join(
            f"{f},{t},{x},{y},{w},{h},1,-1,-1,-1" for f, t, x, y, w, h in rs) + "\n")
        (trk_dir / f"{seq}.txt").write_text("\n".join(
            f"{f},{t},{x},{y},{w},{h},1.0" for f, t, x, y, w, h in rs) + "\n")
        stats[seq] = (len(rs), len({t for _, t, *_ in rs}))
    return stats


def tracker_stats(trk_dir: Path, seqs: set[str]) -> dict:
    out = {}
    for f in sorted(trk_dir.glob("*.txt")):
        if f.stem not in seqs:
            continue
        ids, n = set(), 0
        for line in f.read_text().splitlines():
            if line.strip():
                ids.add(int(float(line.split(",")[1])))
                n += 1
        out[f.stem] = (n, len(ids))
    return out


def run(predictor: str, tracks: Path, pred_out: Path, gt: Path,
        imu_dir: str | None = None) -> dict:
    extra = ["--imu-dir", imu_dir] if imu_dir else []
    subprocess.run(
        [sys.executable, "-m", "prediction.infer",
         "--tracks", str(tracks), "--out", str(pred_out),
         "--model", predictor, "--gpu", "-1", *extra],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    from evaluation.so_hota_tp import evaluate_split

    r = evaluate_split(pred_out, gt, tracks)
    return {
        "SO-HOTA-TP": float(r["SO-HOTA-TP"]),
        "minADE": float(r["minADE"]),
        "minFDE": float(r["minFDE"]),
        "n_windows": int(r["n_windows"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="GT vs real tracker input")
    p.add_argument("--tracker-dir", required=True,
                   help="tracker output, one {seq}.txt per sequence")
    p.add_argument("--annotations", default="SMOT4SB/annotations/train.json")
    p.add_argument("--sequences", nargs="*", default=None,
                   help="limit to these; default is every sequence present")
    p.add_argument("--predictors", nargs="*", default=list(PREDICTORS))
    p.add_argument("--out", default=None, help="write results as JSON")
    p.add_argument("--imu-dir", default=None,
                   help="split directory with {seq}/imu.csv, for checkpoints trained "
                        "with --cmc (their CMC config comes from $SMOTP4SB_CMC_CONFIG)")
    a = p.parse_args()

    trk = Path(a.tracker_dir)
    if not trk.is_dir():
        raise SystemExit(f"no such directory: {trk}")
    seqs = set(a.sequences) if a.sequences else {f.stem for f in trk.glob("*.txt")}
    if not seqs:
        raise SystemExit(f"no .txt files in {trk}")

    tmp = Path(tempfile.mkdtemp())
    try:
        gt_dir, gt_trk = tmp / "gt", tmp / "trk_gt"
        gt_counts = write_gt_arms(ROOT / a.annotations, seqs, gt_dir, gt_trk)
        # Only sequences with GT can be scored at all.
        seqs &= set(gt_counts)
        det_counts = tracker_stats(trk, seqs)

        print(f"{len(seqs)} sequence(s)\n")
        print(f"{'seq':>6} {'GT boxes':>9} {'GT ids':>7} "
              f"{'det boxes':>10} {'det ids':>8} {'id ratio':>9}")
        for s in sorted(seqs):
            gb, gi = gt_counts[s]
            db, di = det_counts.get(s, (0, 0))
            ratio = f"{di / gi:>8.1f}x" if gi else " " * 9
            print(f"{s:>6} {gb:>9,} {gi:>7} {db:>10,} {di:>8} {ratio}")

        results = {}
        print(f"\n{'predictor':>10} {'input':>8} {'SO-HOTA-TP':>11} "
              f"{'minADE':>9} {'minFDE':>9} {'windows':>8}")
        for name in a.predictors:
            results[name] = {}
            for arm, tracks in (("GT", gt_trk), ("tracker", trk)):
                r = run(name, tracks, tmp / f"pred_{name}_{arm}", gt_dir, a.imu_dir)
                results[name][arm] = r
                ade = f"{r['minADE']:9.2f}" if r["minADE"] == r["minADE"] else "      nan"
                print(f"{name:>10} {arm:>8} {r['SO-HOTA-TP']:>11.4f} {ade} "
                      f"{r['minFDE']:>9.2f} {r['n_windows']:>8}")

        print("\nminADE penalty for using a real tracker:")
        for name, arms in results.items():
            g, t = arms["GT"]["minADE"], arms["tracker"]["minADE"]
            if g == g and t == t and g > 0:
                print(f"  {name:>10}  {g:8.2f} -> {t:8.2f}   {(t / g - 1) * 100:+.0f}%")

        total_gi = sum(gi for _, gi in gt_counts.values())
        total_di = sum(di for _, di in det_counts.values())
        print(f"\nIdentity fragmentation: {total_di} tracker ids against "
              f"{total_gi} in the GT ({total_di / max(total_gi, 1):.1f}x)")

        if a.out:
            out = Path(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "sequences": sorted(seqs),
                "gt_counts": {k: list(v) for k, v in gt_counts.items()},
                "tracker_counts": {k: list(v) for k, v in det_counts.items()},
                "results": results,
            }, indent=2))
            print(f"\nwrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
