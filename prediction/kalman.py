"""Kalman-filter baselines for SMOTP4SB trajectory prediction.

A constant-velocity Kalman filter is run over the 60 observed frames, then the
state is propagated 150 frames forward. Two things distinguish it from the
motion models in `motion_models.py`:

1. The velocity estimate is smoothed over the whole observation window rather
   than fitted to its tail, so a single noisy detection cannot hijack it.
2. The filter carries a covariance, so the K modes can be drawn from the
   posterior instead of being K identical copies. This is the only baseline
   here for which minADE_K < ADE_1 -- i.e. the only one that actually uses K.

The transition also supports a velocity damping factor. damp=1.0 is textbook
constant velocity; damp<1 decays speed toward zero, interpolating between CV
and the 'stay' predictor, which matters because bird motion over a 5 s horizon
turns out to be barely persistent.

Usage:
    python -m prediction.kalman
"""

import argparse

import numpy as np

from prediction import paths
from prediction.motion_models import load_val


def kalman_filter(obs: np.ndarray, q: float, r: float, damp: float):
    """Run a CV Kalman filter over obs (B, T, 2). Returns posterior mean (B,4) and cov (B,4,4)."""
    B, T, _ = obs.shape
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, damp, 0], [0, 0, 0, damp]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    Q = np.diag([q, q, q, q]).astype(np.float64)
    R = (np.eye(2) * r).astype(np.float64)

    # Initialise from the first two points; large initial covariance keeps the
    # filter from trusting that seed.
    x = np.zeros((B, 4))
    x[:, :2] = obs[:, 1]
    x[:, 2:] = obs[:, 1] - obs[:, 0]
    P = np.tile(np.diag([r, r, 100.0, 100.0]), (B, 1, 1))

    for t in range(2, T):
        x = x @ F.T
        P = F @ P @ F.T + Q
        y = obs[:, t] - x @ H.T                        # innovation
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)                 # (B,4,2)
        x = x + np.einsum('bij,bj->bi', K, y)
        P = P - K @ (H @ P)
    return x, P


def propagate(state: np.ndarray, t_pred: int, damp: float) -> np.ndarray:
    """Roll the state forward t_pred steps, returning positions (B, t_pred, 2)."""
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    out = np.empty((len(state), t_pred, 2))
    for t in range(t_pred):
        pos = pos + vel
        vel = vel * damp
        out[:, t] = pos
    return out


def sampled_modes(x, P, t_pred, k, damp, spread, seed=0):
    """Draw K states from the posterior and propagate each -> (B, K, t_pred, 2).

    `spread` scales the sampling covariance. The filter's own P reflects
    measurement noise, not how much a bird may manoeuvre over 5 s, so it
    understates the spread the modes need to cover.
    """
    rng = np.random.default_rng(seed)
    B = len(x)
    # Symmetrise before the Cholesky: repeated updates leave tiny asymmetries.
    Ps = 0.5 * (P + np.transpose(P, (0, 2, 1))) * spread + np.eye(4) * 1e-6
    L = np.linalg.cholesky(Ps)
    out = np.empty((B, k, t_pred, 2))
    for i in range(k):
        z = rng.standard_normal((B, 4))
        out[:, i] = propagate(x + np.einsum('bij,bj->bi', L, z), t_pred, damp)
    return out


def minade(modes: np.ndarray, gt: np.ndarray, k: int, thr: float) -> dict:
    d = np.linalg.norm(modes - gt[:, None], axis=-1)      # (B,K,T)
    ade, fde = d.mean(axis=2), d[:, :, -1]
    best = ade.argmin(axis=1)
    rows = np.arange(len(gt))
    return {f"minADE_{k}": float(ade.min(axis=1).mean()),
            f"minFDE_{k}": float(fde.min(axis=1).mean()),
            f"MissRate_{k}@{thr:g}px": float((fde[rows, best] > thr).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description="Kalman-filter trajectory baselines")
    p.add_argument("--t-obs", type=int, default=60)
    p.add_argument("--t-pred", type=int, default=150)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--miss-threshold", type=float, default=48.0)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()

    obs, gt = load_val(a.t_obs, a.t_pred, a.split, a.stride)
    obs = obs.astype(np.float64)
    K, thr = a.k, a.miss_threshold
    print(f"val windows: {len(obs)}\n")
    print(f"  {'model':<34}{'minADE_%d' % K:>12}{'minFDE_%d' % K:>13}{'MR@%g' % thr:>10}")

    def row(name, modes):
        m = minade(modes, gt, K, thr)
        print(f"  {name:<34}{m[f'minADE_{K}']:>12.2f}{m[f'minFDE_{K}']:>13.2f}"
              f"{m[f'MissRate_{K}@{thr:g}px']:>10.3f}")

    row("stay", np.repeat(np.repeat(obs[:, -1:, :], a.t_pred, 1)[:, None], K, 1))

    for damp in (1.0, 0.99, 0.95, 0.9):
        x, P = kalman_filter(obs, q=1.0, r=4.0, damp=damp)
        mean = propagate(x, a.t_pred, damp)
        row(f"KF mean (damp={damp})", np.repeat(mean[:, None], K, 1))

    # Modes drawn from the posterior: the only baseline where K>1 does work.
    # The reported baseline is (damp=0.95, s=10) -- marked below. s=20 takes
    # minADE 1.0 px lower but misses far more often (MR 0.896 against 0.857),
    # which is the wrong trade for a baseline meant to be reliable.
    for damp in (1.0, 0.95):
        for spread in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0):
            x, P = kalman_filter(obs, q=1.0, r=4.0, damp=damp)
            mark = "  <- reported" if (damp, spread) == (0.95, 10.0) else ""
            row(f"KF sampled K={K} (damp={damp}, s={spread:g}){mark}",
                sampled_modes(x, P, a.t_pred, K, damp, spread))


if __name__ == "__main__":
    main()
