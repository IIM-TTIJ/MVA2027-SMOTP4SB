"""Remove and re-apply camera rotation on pixel trajectories.

For a camera undergoing pure rotation, the mapping between two frames is a
homography built from the two orientations and the intrinsics (guide 5.2):

    H(t -> ref) = K @ R_ref @ inv(R_t) @ K_inv

Applying it to a pixel at frame t gives where that world point would have landed
had the camera never moved from its reference orientation. Undoing it is the
same expression with the two orientations swapped:

    H(ref -> t) = K @ R_t @ inv(R_ref) @ K_inv

The two compose to the identity exactly, which the round-trip test checks.

Reference frame is the **first observed frame** of the window.
Both directions must use the same reference, so `inverse_compensate` takes it
explicitly rather than guessing from the future rotations it is handed.

Pure rotation is an approximation: it ignores parallax from camera translation.
That holds here because the birds are far away relative to how far the drone
moves between frames, and it is the standard assumption in aerial stabilisation.
"""

import numpy as np


def apply_homographies(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a per-frame homography stack to a batch of pixel trajectories.

    points: (..., T, 2)      H: (T, 3, 3)      -> (..., T, 2)

    Mathematically identical to the per-point reference loop in docs/cmc.md; written as an einsum
    because this runs for every mode of every window, where a Python loop costs
    real time. tests/test_cmc.py checks it against a literal transcription of
    that loop.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError(f"trajectory must end in an (x, y) axis, got {points.shape}")
    T = points.shape[-2]
    if H.shape != (T, 3, 3):
        raise ValueError(f"need one homography per frame: points have T={T}, H is {H.shape}")

    homogeneous = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
    warped = np.einsum("tij,...tj->...ti", H, homogeneous)

    w = warped[..., 2]
    # w collapses to zero only if a point maps to the plane at infinity, i.e. the
    # camera rotated so far that the point left the projective frame. Silently
    # dividing would emit inf and poison the metric downstream.
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("homography sent a point to infinity; rotation is implausibly large")
    return warped[..., :2] / w[..., None]


def compensate_trajectory(trajectory: np.ndarray, rotations: np.ndarray,
                          K: np.ndarray, ref_index: int = 0) -> np.ndarray:
    """Map a pixel trajectory into the reference frame, removing camera motion.

    trajectory: (..., T, 2) pixels        rotations: (T, 3, 3) world-to-camera
    ref_index:  which frame's orientation to use as reference (default 0).
    Returns (..., T, 2) in the coordinates of the reference frame's orientation.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    R_ref = rotations[ref_index]
    H = K @ R_ref @ np.linalg.inv(rotations) @ K_inv
    return apply_homographies(trajectory, H)


def inverse_compensate(pred_trajectory: np.ndarray, future_rotations: np.ndarray,
                       K: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Map a compensated prediction back into raw pixel coordinates.

    pred_trajectory: (..., T, 2) in reference-frame coordinates -- typically
        (K_modes, T_pred, 2) straight out of the model.
    future_rotations: (T, 3, 3) camera orientation at each predicted frame.
    R_ref: the reference orientation used when compensating the observation,
        i.e. the first observed frame. Passing anything else silently shifts
        every prediction, so it is required rather than inferred.
    """
    future_rotations = np.asarray(future_rotations, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R_ref = np.asarray(R_ref, dtype=np.float64)
    if R_ref.shape != (3, 3):
        raise ValueError(f"R_ref must be a single (3, 3) rotation, got {R_ref.shape}")
    K_inv = np.linalg.inv(K)
    H_inv = K @ future_rotations @ np.linalg.inv(R_ref) @ K_inv
    return apply_homographies(pred_trajectory, H_inv)
