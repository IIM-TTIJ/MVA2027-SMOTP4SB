"""Build SMOTP4SB trajectory-prediction windows from SMOT4SB GT annotations.

Extracts gap-free sliding windows of T_OBS + T_PRED frames from the MOT track
annotations, splits them by sequence (never by window -- overlapping windows
from one track would leak across the split), and saves them as .npz.

Per the Track 2 guide: training uses stride=1 to maximise data, evaluation uses
non-overlapping windows (stride = T_OBS + T_PRED). Windows containing detection
gaps are excluded rather than interpolated.

Usage:
    python -m prediction.build_dataset --t-obs 60 --t-pred 150 --out data/traj
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from prediction import paths


def load_tracks() -> dict:
    """Return {(sequence, track_id): (frames[N], xy[N, 2])}, frame-sorted.

    track_id is only unique within a video, so the sequence name is part of the key.
    Trajectories are bounding-box centres.
    """
    ann = paths.require(
        paths.ANNOTATIONS, "SMOT4SB annotations",
        "Symlink the dataset as <repo>/SMOT4SB, or set SMOT4SB_ROOT=/path/to/SMOT4SB")
    data = json.loads(ann.read_text())
    images = {i["id"]: i for i in data["images"]}
    raw = collections.defaultdict(list)
    for a in data["annotations"]:
        im = images[a["image_id"]]
        seq = im["file_name"].split("/")[0]
        x, y, w, h = a["bbox"]
        raw[(seq, a["track_id"])].append((im["frame_id"], x + w / 2.0, y + h / 2.0))

    tracks = {}
    for key, pts in raw.items():
        pts.sort()
        frames = np.array([p[0] for p in pts], dtype=np.int64)
        xy = np.array([(p[1], p[2]) for p in pts], dtype=np.float32)
        tracks[key] = (frames, xy)
    return tracks


def gap_free_windows(frames: np.ndarray, win: int, stride: int) -> list[int]:
    """Start indices of length-`win` runs of consecutive frame numbers."""
    if len(frames) < win:
        return []
    # run[i] = length of the consecutive run ending at i
    run = np.ones(len(frames), dtype=np.int64)
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            run[i] = run[i - 1] + 1
    ends = np.flatnonzero(run >= win)
    starts = sorted(int(e - win + 1) for e in ends)
    if stride == 1:
        return starts
    # Step by `stride` from each accepted start; stride == win gives
    # non-overlapping windows, anything smaller gives controlled overlap.
    kept, next_ok = [], -1
    for s in starts:
        if s >= next_ok:
            kept.append(s)
            next_ok = s + stride
    return kept


def build(t_obs: int, t_pred: int, val_every: int, out: Path) -> None:
    win = t_obs + t_pred
    tracks = load_tracks()
    sequences = sorted({seq for seq, _ in tracks})
    # Hold out every Nth sequence, so whole videos (not windows) are unseen.
    val_seqs = set(sequences[::val_every])
    print(f"sequences: {len(sequences)} total -> {len(sequences)-len(val_seqs)} train / {len(val_seqs)} val")

    split_data = {"train": [], "val": []}
    split_meta = {"train": [], "val": []}
    for (seq, tid), (frames, xy) in sorted(tracks.items()):
        is_val = seq in val_seqs
        split = "val" if is_val else "train"
        # stride=1 for training, non-overlapping for evaluation
        for s in gap_free_windows(frames, win, win if is_val else 1):
            split_data[split].append(xy[s:s + win])
            split_meta[split].append((int(seq), tid, int(frames[s])))

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        arr = np.asarray(split_data[split], dtype=np.float32)
        meta = np.asarray(split_meta[split], dtype=np.int64)
        np.savez_compressed(out / f"{split}.npz", traj=arr, meta=meta,
                            t_obs=t_obs, t_pred=t_pred)
        print(f"  {split}: {arr.shape[0]} windows, array {arr.shape}")
    (out / "split.json").write_text(json.dumps(
        {"val_sequences": sorted(val_seqs), "t_obs": t_obs, "t_pred": t_pred}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Build trajectory windows from SMOT4SB GT")
    p.add_argument("--t-obs", type=int, default=60)
    p.add_argument("--t-pred", type=int, default=150)
    p.add_argument("--val-every", type=int, default=5, help="hold out every Nth sequence")
    p.add_argument("--out", type=str, default=str(paths.TRAJ),
                   help="output directory (relative paths are taken from the repo root)")
    a = p.parse_args()
    build(a.t_obs, a.t_pred, a.val_every, paths.resolve(a.out))


if __name__ == "__main__":
    main()
