"""Predict camera orientation for the frames we have to forecast.

Compensating the observation needs measured rotations, which we have. Undoing
compensation on the *prediction* needs rotations for the 150 future frames,
which we do not: IMU data covers the observation window only. They have to be
extrapolated, and that extrapolation is its own error source -- separate from
how well the model predicts bird motion.

Keeping the two apart matters. If a CMC result disappoints, the cause is either
bad future-rotation prediction or bad trajectory prediction, and only an oracle
(guide B2) separates them.

Usage:
    from cmc.extrapolate import extrapolate
    R_future = extrapolate(R_observed, n_future=150)
"""

import numpy as np
from scipy.spatial.transform import Rotation


def constant_omega(R_obs: np.ndarray, n_future: int) -> np.ndarray:
    """Assume the camera keeps turning at the rate seen in the last two frames.

    Takes the frame-to-frame increment dR = R[-1] @ inv(R[-2]), converts it to
    axis-angle, and scales the angle by k to get dR^k -- exact for a rotation,
    since powers of a rotation share its axis and multiply its angle.

        R_future[k] = (dR^k) @ R[-1]      for k = 1 .. n_future

    Cheap and sufficient for a baseline. Its weakness is that a
    single noisy pair of frames sets the rate for all 150 predicted frames; the
    'uniform' mode exists to average over a longer window instead.
    """
    R_obs = np.asarray(R_obs, dtype=np.float64)
    if len(R_obs) < 2:
        raise ValueError("constant_omega needs at least 2 observed rotations")
    dR = R_obs[-1] @ np.linalg.inv(R_obs[-2])
    rotvec = Rotation.from_matrix(dR).as_rotvec()          # axis * angle, one frame
    k = np.arange(1, n_future + 1, dtype=np.float64)
    steps = Rotation.from_rotvec(k[:, None] * rotvec)      # dR^k for every k
    return steps.as_matrix() @ R_obs[-1]


def uniform(R_obs: np.ndarray, n_future: int) -> np.ndarray:
    """Average the turn rate over the whole observation window, then hold it.

    UCMCTrack (AAAI 2024) found that one constant compensation across a sequence
    beat noisy per-frame estimates. Here that means deriving the rate from the
    full window rather than its last two frames: the total rotation across the
    window, divided by the number of steps.
    """
    R_obs = np.asarray(R_obs, dtype=np.float64)
    if len(R_obs) < 2:
        raise ValueError("uniform needs at least 2 observed rotations")
    total = R_obs[-1] @ np.linalg.inv(R_obs[0])
    per_step = Rotation.from_matrix(total).as_rotvec() / (len(R_obs) - 1)
    k = np.arange(1, n_future + 1, dtype=np.float64)
    return Rotation.from_rotvec(k[:, None] * per_step).as_matrix() @ R_obs[-1]


def oracle(R_obs: np.ndarray, n_future: int, R_true_future: np.ndarray) -> np.ndarray:
    """Use the true future rotations. Not a predictor -- an upper bound.

    Reporting this next to a real mode separates 'CMC does not help' from 'our
    future-rotation guess is bad'. With synthetic data the truth is free; with
    real logs it means holding out the tail of a log we already have.
    """
    R_true_future = np.asarray(R_true_future, dtype=np.float64)
    if len(R_true_future) < n_future:
        raise ValueError(f"oracle needs {n_future} future rotations, "
                         f"got {len(R_true_future)}")
    return R_true_future[:n_future]


MODES = {"constant_omega": constant_omega, "uniform": uniform}


def extrapolate(R_obs: np.ndarray, n_future: int, mode: str = "constant_omega",
                R_true_future: np.ndarray | None = None) -> np.ndarray:
    """Dispatch to an extrapolation mode. Returns (n_future, 3, 3)."""
    if mode == "oracle":
        if R_true_future is None:
            raise ValueError("mode='oracle' requires R_true_future")
        return oracle(R_obs, n_future, R_true_future)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from "
                         f"{sorted(MODES) + ['oracle']}")
    return MODES[mode](R_obs, n_future)
