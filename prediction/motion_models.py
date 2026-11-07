"""Training-free motion-model baselines for SMOTP4SB trajectory prediction.

Implements the classical single-object motion models on bird tracks:

  stay  - repeat the last observed position
  CV    - constant velocity
  CTRV  - constant turn rate and velocity (circular arc, constant speed)
  CTRA  - constant turn rate and acceleration (circular arc, changing speed)

All are deterministic, so every one of the K modes is identical and
minADE_K == ADE_1. They exist to establish the floor a learned multimodal model
must clear.

Parameters are estimated from the tail of the observation window rather than the
whole of it: over 60 frames a bird's heading changes enough that an average over
all of it describes no part of the motion. Propagation is a step-by-step
integration, which avoids the 1/omega^2 singularity in the closed-form CTRA
equations as the turn rate approaches zero.

Usage:
    python -m prediction.motion_models          # per-model tuned windows
"""

import argparse
import json

import numpy as np

from prediction import paths
from prediction.build_dataset import gap_free_windows, load_tracks
from prediction.eval_constant_velocity import metrics


def _estimate(obs: np.ndarray, w: int):
    """Least-squares estimate of speed, heading, turn rate and acceleration.

    Fits heading and speed linearly over the last w steps and reads both the
    value at the final frame and the slope off the fit. Taking either from the
    single last step instead makes them hostage to one noisy detection, which
    then gets extrapolated for 150 frames.
    """
    tail = obs[:, -(w + 1):, :]
    d = np.diff(tail, axis=1)                                  # (B, w, 2) per-frame steps
    speed = np.linalg.norm(d, axis=2)                          # (B, w)
    heading = np.arctan2(d[:, :, 1], d[:, :, 0])               # (B, w)
    # Unwrap along time so a track crossing the +/-pi branch cut does not
    # register a full-turn jump.
    heading = np.unwrap(heading, axis=1)

    n = heading.shape[1]
    if n < 2:
        return speed[:, -1], heading[:, -1], np.zeros(len(obs)), np.zeros(len(obs))
    t = np.arange(n, dtype=np.float32)
    tc = t - t.mean()
    denom = (tc ** 2).sum()

    def fit(y):
        """Return (value at final step, slope per step)."""
        slope = ((y - y.mean(axis=1, keepdims=True)) * tc).sum(axis=1) / denom
        intercept = y.mean(axis=1) - slope * t.mean()
        return intercept + slope * t[-1], slope

    theta0, omega = fit(heading)
    v0, accel = fit(speed)
    return np.maximum(v0, 0.0), theta0, omega, accel


def propagate(obs, t_pred, w=20, turn=True, accel=True, damp=1.0):
    """Integrate a CTRA-family model forward t_pred steps.

    turn=False, accel=False -> constant velocity
    turn=True,  accel=False -> CTRV
    turn=True,  accel=True  -> CTRA
    damp scales omega and a, to test whether weaker extrapolation helps.
    """
    v, theta, omega, a = _estimate(obs, w)
    if not turn:
        omega = np.zeros_like(omega)
    if not accel:
        a = np.zeros_like(a)
    omega, a = omega * damp, a * damp

    pos = obs[:, -1, :].copy()
    out = np.empty((len(obs), t_pred, 2), dtype=np.float32)
    for t in range(t_pred):
        theta = theta + omega
        v = v + a
        # A bird cannot fly backwards along its own heading; without this the
        # acceleration term drags tracks into reverse and errors explode.
        v = np.maximum(v, 0.0)
        pos = pos + np.stack([v * np.cos(theta), v * np.sin(theta)], axis=1)
        out[:, t, :] = pos
    return out


def load_val(t_obs, t_pred, split, stride):
    val = set(json.load(open(paths.resolve(split)))["val_sequences"])
    win = t_obs + t_pred
    w = [xy[s:s + win] for (q, _), (f, xy) in sorted(load_tracks().items())
         if q in val for s in gap_free_windows(f, win, stride)]
    d = np.asarray(w, dtype=np.float32)
    return d[:, :t_obs], d[:, t_obs:]


def main() -> None:
    p = argparse.ArgumentParser(description="Motion-model baselines")
    p.add_argument("--t-obs", type=int, default=60)
    p.add_argument("--t-pred", type=int, default=150)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--miss-threshold", type=float, default=48.0)
    p.add_argument("--est-window", type=int, default=None,
                   help="force one estimation window for every model; by default "
                        "each uses its own tuned value (CV/CTRA 10, CTRV 3)")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()

    obs, gt = load_val(a.t_obs, a.t_pred, a.split, a.stride)

    # Each model's estimation window is tuned separately. CTRV needs a far
    # shorter tail than CV: a turn rate averaged over ten frames of a banking
    # bird describes no part of the turn, and forcing one shared window
    # reproduces none of the reported numbers.
    TUNED = {"CV": 10, "CTRV": 3, "CTRA": 10}
    def w(name):
        return a.est_window if a.est_window else TUNED[name]

    # This CV row estimates velocity by least squares, so CV / CTRV / CTRA share
    # one estimator and stay comparable. It is NOT the reference constant velocity,
    # which extrapolates p[-1] - p[-2] -- see
    # prediction/eval_constant_velocity.py.
    models = [
        ("stay", np.repeat(obs[:, -1:, :], a.t_pred, axis=1)),
        (f"CV (LS fit, w={w('CV')})",
         propagate(obs, a.t_pred, w("CV"), turn=False, accel=False)),
        (f"CTRV (w={w('CTRV')})",
         propagate(obs, a.t_pred, w("CTRV"), turn=True, accel=False)),
        (f"CTRA (w={w('CTRA')})",
         propagate(obs, a.t_pred, w("CTRA"), turn=True, accel=True)),
        ("CTRV damped 0.3",
         propagate(obs, a.t_pred, w("CTRV"), turn=True, accel=False, damp=0.3)),
        ("CTRA damped 0.1",
         propagate(obs, a.t_pred, w("CTRA"), turn=True, accel=True, damp=0.1)),
    ]
    tag = f"forced window {a.est_window}" if a.est_window else "per-model tuned windows"
    print(f"val windows: {len(obs)}   {tag}\n")
    print(f"  {'model':<22}{'minADE_%d' % a.k:>12}{'minFDE_%d' % a.k:>13}{'MR@%g' % a.miss_threshold:>10}")
    for name, pred in models:
        m = metrics(pred.astype(np.float32), gt, a.k, a.miss_threshold)
        print(f"  {name:<22}{m[f'minADE_{a.k}']:>12.2f}{m[f'minFDE_{a.k}']:>13.2f}"
              f"{m[f'MissRate_{a.k}@{a.miss_threshold:g}px']:>10.3f}")


if __name__ == "__main__":
    main()
