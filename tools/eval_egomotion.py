"""How much camera motion does IMU-based CMC remove? Measured on the frames.

For each sequence with an imu.csv, 60-frame windows (stride 30) are compensated
into their last frame -- the Track 2 input_only setting -- with
  * the true camera motion, measured on the frames (cmc/measure.py), and
  * the motion the egomotion model predicts from the window's imu.csv rows.
The error is the mean distance between where the two put a 5x5 grid of image
points, over all frames of the window. "none" is no compensation. The printed
reduction is how much of the camera-induced displacement the IMU removes.

Usage:
    python tools/eval_egomotion.py --split pub_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cmc.config import CMCConfig  # noqa: E402
from cmc.egomotion import EgoMotionModel, homographies_to_ref, imu_window, load_imu  # noqa: E402
from cmc.compensate import apply_homographies  # noqa: E402
from cmc.intrinsics import K_for  # noqa: E402
from cmc.pipeline import has_motion_source, imu_file  # noqa: E402
from cmc.measure import cached_motion, clean_motion  # noqa: E402
from prediction import paths  # noqa: E402

WINDOW = 60


def grid(w: int, h: int) -> np.ndarray:
    return np.array([(x, y) for x in np.linspace(0.1 * w, 0.9 * w, 5) for y in np.linspace(0.1 * h, 0.9 * h, 5)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="pub_test")
    ap.add_argument("--model", default=str(ROOT / "cmc" / "egomotion_model.json"))
    ap.add_argument("--motion-cache", default=str(paths.DATA / "egomotion_motion"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = CMCConfig(split=a.split)
    model = EgoMotionModel.load(a.model)
    root = paths.SMOT4SB / a.split
    seqs = sorted(p.name for p in root.iterdir() if p.is_dir() and has_motion_source(p.name, cfg))
    leaked = [q for q in seqs if f"{a.split}/{q}" in model.trained_on]
    if leaked:
        print(f"  warning: the model was fitted on {leaked}")
    tot_m, tot_0 = [], []
    for q in seqs:
        imu = load_imu(imu_file(q, cfg))
        K = K_for(q, cfg); cx, cy = float(K[0, 2]), float(K[1, 2])
        w, h = round(2 * cx), round(2 * cy)
        true, _ = clean_motion(cached_motion(root / q, a.motion_cache, f"{a.split}/{q}", a.workers), w)
        G = grid(w, h)
        em, e0 = [], []
        for s in range(0, len(imu["frame"]) - WINDOW + 1, 30):
            pred = model.predict_imu(imu_window(imu, s + 1, WINDOW), width=w)
            Ht = homographies_to_ref(true[s:s + WINDOW - 1], WINDOW - 1, (cx, cy))
            Hp = homographies_to_ref(pred, WINDOW - 1, (cx, cy))
            P = np.broadcast_to(G, (WINDOW,) + G.shape).transpose(1, 0, 2)        # (points, T, 2)
            At = apply_homographies(P, Ht); Ap = apply_homographies(P, Hp)
            em.append(float(np.linalg.norm(At - Ap, axis=-1).mean()))
            e0.append(float(np.linalg.norm(At - P, axis=-1).mean()))
        if em:
            print(f"  {a.split}/{q}: {len(em)} windows, IMU CMC {np.mean(em):7.2f} px, none {np.mean(e0):7.2f} px")
            tot_m += em; tot_0 += e0
    if not tot_m:
        raise SystemExit("no windows evaluated")
    m, z = float(np.mean(tot_m)), float(np.mean(tot_0))
    print(f"{a.split}: {len(tot_m)} windows; residual camera displacement {m:.2f} px with IMU CMC, "
          f"{z:.2f} px without ({100 * (1 - m / z):.1f}% removed)")


if __name__ == "__main__":
    main()
