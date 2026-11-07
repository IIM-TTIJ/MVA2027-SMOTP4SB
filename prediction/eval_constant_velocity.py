"""Constant-velocity baseline for SMOTP4SB trajectory prediction.

A training-free reference for Track 2: extrapolate
linearly from the observed trajectory and replicate the result K times. Because
the predictor is deterministic all K modes are identical, so minADE_K equals
ADE_1 -- that is the point of the baseline, and any real multimodal model must
beat it to justify its modes.

Reports minADE_K, minFDE_K and Miss Rate at the task's 48 px threshold.

Usage:
    python -m prediction.eval_constant_velocity --t-obs 60 --t-pred 150 --k 6
"""

import argparse
import json

import numpy as np

from prediction import paths
from prediction.build_dataset import gap_free_windows, load_tracks


def cv_predict(obs: np.ndarray, t_pred: int, smooth: int) -> np.ndarray:
    """Extrapolate at constant velocity from the last `smooth`+1 observations.

    obs: (B, T_OBS, 2) -> (B, T_PRED, 2)
    smooth=1 reproduces the "last 2 positions" rule; larger windows
    average the velocity over more frames, which damps tracker jitter.
    """
    last = obs[:, -1, :]
    span = min(smooth, obs.shape[1] - 1)
    vel = (obs[:, -1, :] - obs[:, -1 - span, :]) / span
    steps = np.arange(1, t_pred + 1, dtype=np.float32)[None, :, None]
    return last[:, None, :] + vel[:, None, :] * steps


def metrics(pred: np.ndarray, gt: np.ndarray, k: int, miss_thresh: float) -> dict:
    """pred: (B, T, 2) single mode -> replicated to K. gt: (B, T, 2)."""
    modes = np.repeat(pred[:, None], k, axis=1)          # (B, K, T, 2)
    d = np.linalg.norm(modes - gt[:, None], axis=-1)     # (B, K, T)
    ade = d.mean(axis=2)                                 # (B, K)
    fde = d[:, :, -1]                                    # (B, K)
    best = ade.argmin(axis=1)
    rows = np.arange(len(pred))
    return {
        f"minADE_{k}": float(ade.min(axis=1).mean()),
        f"minFDE_{k}": float(fde.min(axis=1).mean()),
        f"MissRate_{k}@{miss_thresh:g}px": float((fde[rows, best] > miss_thresh).mean()),
        "n_windows": int(len(pred)),
    }


def collect(t_obs: int, t_pred: int, val_seqs: set, stride: int):
    win = t_obs + t_pred
    out = []
    for (seq, _), (frames, xy) in sorted(load_tracks().items()):
        if seq not in val_seqs:
            continue
        for s in gap_free_windows(frames, win, stride):
            out.append(xy[s:s + win])
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Constant-velocity trajectory baseline")
    p.add_argument("--t-obs", type=int, default=60)
    p.add_argument("--t-pred", type=int, default=150)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--miss-threshold", type=float, default=48.0)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()

    val_seqs = set(json.load(open(paths.resolve(a.split)))["val_sequences"])
    win = a.t_obs + a.t_pred

    for label, stride in (("official (non-overlapping)", win), ("dense (stride 30)", 30)):
        data = collect(a.t_obs, a.t_pred, val_seqs, stride)
        if not len(data):
            print(f"\n{label}: no windows")
            continue
        obs, gt = data[:, :a.t_obs], data[:, a.t_obs:]
        print(f"\n=== {label} ===")
        for smooth in (1, 5, 15):
            m = metrics(cv_predict(obs, a.t_pred, smooth), gt, a.k, a.miss_threshold)
            tag = "last-2 (guide spec)" if smooth == 1 else f"vel over {smooth} frames"
            print(f"  {tag:<22} minADE_{a.k}={m[f'minADE_{a.k}']:8.2f} px   "
                  f"minFDE_{a.k}={m[f'minFDE_{a.k}']:9.2f} px   "
                  f"MR@{a.miss_threshold:g}={m[f'MissRate_{a.k}@{a.miss_threshold:g}px']:.3f}   "
                  f"n={m['n_windows']}")


if __name__ == "__main__":
    main()
