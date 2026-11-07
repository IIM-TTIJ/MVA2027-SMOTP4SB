"""Fit the IMU -> camera-motion model used by CMC (cmc/egomotion.py).

For every training sequence that has an imu.csv:
  1. measure the camera's motion on its frames (cmc/measure.py), cached per
     sequence because it is the slow part;
  2. cut 60-frame windows (stride 10) and build each window's features from its
     own imu.csv rows only, so the lags are truncated exactly as at run time,
     where Track 2 compensates the 60 observed frames;
  3. fit a ridge regression to the measured motion, ignoring transitions whose
     measurement failed or is implausibly large, and store a shrink factor for
     its predictions (--gain): errors accumulate over a window, so a damped
     prediction removes more camera motion than the raw one.

Usage:
    python tools/fit_egomotion.py --out cmc/egomotion_model.json
    python tools/fit_egomotion.py --split-json data/traj/split.json --exclude-val ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cmc.config import CMCConfig  # noqa: E402
from cmc.egomotion import EgoMotionModel, features, imu_window, load_imu  # noqa: E402
from cmc.measure import REF_WIDTH, cached_motion, clean_motion  # noqa: E402
from cmc.pipeline import has_motion_source, imu_file  # noqa: E402
from prediction import paths  # noqa: E402

WINDOW = 60
STRIDE = 10


def window_samples(imu: dict, motion: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, targets (scaled to REF_WIDTH) and weights for every window."""
    T = len(imu["frame"])
    if len(motion) != T - 1:
        raise ValueError(f"{T} imu rows but {len(motion)} measured transitions")
    Y, bad = clean_motion(motion, width)
    Y[:, :2] *= REF_WIDTH / width
    Xs, Ys, Ws = [], [], []
    for s in range(0, max(T - WINDOW, 0) + 1, STRIDE):
        n = min(WINDOW, T - s)
        if n < 2:
            continue
        Xs.append(features(imu_window(imu, s + 1, n)))
        Ys.append(Y[s:s + n - 1]); Ws.append((~bad[s:s + n - 1]).astype(float))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ws)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "cmc" / "egomotion_model.json"))
    ap.add_argument("--split", default="train", help="dataset split to fit on")
    ap.add_argument("--exclude-val", action="store_true",
                    help="leave out the validation sequences listed in --split-json")
    ap.add_argument("--split-json", default=str(paths.SPLIT))
    ap.add_argument("--motion-cache", default=str(paths.DATA / "egomotion_motion"))
    ap.add_argument("--lam", type=float, default=3000.0)
    ap.add_argument("--gain", type=float, default=0.5,
                    help="shrink predictions; 0.5 was best in leave-one-recording-out validation")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cfg = CMCConfig(split=a.split)
    root = paths.SMOT4SB / a.split
    seqs = sorted(p.name for p in root.iterdir() if p.is_dir() and has_motion_source(p.name, cfg))
    if a.exclude_val:
        val = set(json.load(open(paths.resolve(a.split_json)))["val_sequences"])
        seqs = [q for q in seqs if q not in val]
    if not seqs:
        raise SystemExit(f"no sequence in {root} has a rebuilt imu.csv (check SMOTP4SB_IMU_DIR)")
    from cmc.intrinsics import K_for
    X, Y, W = [], [], []
    for q in seqs:
        imu = load_imu(imu_file(q, cfg))
        width = round(2 * float(K_for(q, cfg)[0, 2]))
        motion = cached_motion(root / q, a.motion_cache, f"{a.split}/{q}", a.workers)
        x, y, w = window_samples(imu, motion, width)
        X.append(x); Y.append(y); W.append(w)
        print(f"  {a.split}/{q}: {len(imu['frame'])} frames, {int(w.sum())}/{len(w)} usable window transitions")
    model = EgoMotionModel.fit(np.concatenate(X), np.concatenate(Y), np.concatenate(W), lam=a.lam, gain=a.gain,
                               image_size=(REF_WIDTH, REF_WIDTH * 9 // 16),
                               trained_on=tuple(f"{a.split}/{q}" for q in seqs))
    model.save(a.out)
    print(f"fitted on {len(seqs)} sequences -> {a.out}")


if __name__ == "__main__":
    main()
