"""Wiring CMC into the trajectory pipeline.

Two places touch the data, and they must agree on the reference frame:

  training/observation  compensate the whole 210-frame window against its own
                        first frame, so the model both sees and predicts in
                        reference-frame coordinates.
  evaluation            compensate the observation, let the model predict, then
                        inverse-compensate the K modes back to raw pixels using
                        *extrapolated* future rotations, and only then score
                        against raw-pixel ground truth.

Scoring in compensated space instead would quietly grade the model on an easier
problem than the competition does.

Where the camera motion comes from is cfg.source: "egomotion" (the default)
predicts per-frame image motion from imu.csv with the fitted model of
cmc/egomotion.py; "rotations" builds K R K^-1 from stored rotations. Either way
the result is one homography per frame into the reference frame, and the rest
of this module does not care which.

Everything here is a no-op unless cfg.enabled is true.
"""

import functools
from pathlib import Path

import numpy as np

from cmc.compensate import (apply_homographies, compensate_trajectory,
                            inverse_compensate)
from cmc.config import CMCConfig
from cmc.egomotion import EgoMotionModel, homographies_to_ref, imu_window, load_imu
from cmc.extrapolate import extrapolate
from cmc.intrinsics import K_for
from cmc.rotations import has_rotation_source, load_rotations
from prediction import paths


# --------------------------------------------------------------------------- #
# IMU egomotion source
# --------------------------------------------------------------------------- #
def imu_file(video_id: str, cfg: CMCConfig) -> Path:
    root = cfg.imu_dir if cfg.imu_dir is not None else paths.SMOT4SB
    return Path(root) / cfg.split / str(video_id) / "imu.csv"


@functools.lru_cache(maxsize=256)
def _imu_cached(path: str) -> dict:
    return load_imu(path)


@functools.lru_cache(maxsize=4)
def _model_cached(path: str) -> EgoMotionModel:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"egomotion model not found at {p}\n"
            "  it ships with the repository; refit it with tools/fit_egomotion.py")
    return EgoMotionModel.load(p)


def has_motion_source(video_id: str, cfg: CMCConfig) -> bool:
    """Whether this sequence can be compensated with cfg.source."""
    if cfg.source == "rotations":
        return has_rotation_source(video_id, cfg)
    p = imu_file(video_id, cfg)
    if not p.is_file():
        return False
    with open(p) as fh:
        return "gimbal_yaw_cmd" in fh.readline()


def _ego_homographies(video_id: str, start_frame: int, n_rows: int, n_total: int,
                      ref_index: int, cfg: CMCConfig) -> np.ndarray:
    """(n_total, 3, 3) frame -> reference maps from the IMU rows of n_rows frames.

    Only the rows of frames start_frame .. start_frame+n_rows-1 are read. Frames
    beyond them (n_total > n_rows) assume no camera motion (cfg.future="hold").
    """
    imu = imu_window(_imu_cached(str(imu_file(video_id, cfg))), start_frame, n_rows)
    K = K_for(video_id, cfg)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    motion = _model_cached(str(cfg.egomotion_model)).predict_imu(imu, width=round(2 * cx))
    if n_total > n_rows:
        motion = np.concatenate([motion, np.zeros((n_total - n_rows, 3))])
    return homographies_to_ref(motion, ref_index, (cx, cy))


@functools.lru_cache(maxsize=256)
def _rotations_cached(video_id: str, rotation_dir: str, split: str) -> np.ndarray:
    """One file read per video, not per window -- there are tens of thousands.

    The read also checks the file belongs to this sequence's footage, which is
    why the split is part of the key: the same id is different footage in each.
    """
    cfg = CMCConfig(rotation_dir=Path(rotation_dir), split=split)
    R = load_rotations(video_id, cfg)
    R.flags.writeable = False
    return R


def rotations_for(video_id: str, cfg: CMCConfig) -> np.ndarray:
    return _rotations_cached(str(video_id), str(cfg.rotation_dir), cfg.split)


def _slice(R: np.ndarray, start_frame: int, n: int, video_id: str) -> np.ndarray:
    """Frame numbers are 1-based in the MOT annotations; R is 0-indexed."""
    i = start_frame - 1
    if i < 0 or i + n > len(R):
        raise ValueError(
            f"video {video_id}: need frames {start_frame}..{start_frame + n - 1} "
            f"but only {len(R)} rotations are available")
    return R[i:i + n]


def reference_index(cfg: CMCConfig, t_obs: int) -> int:
    """Which frame of the window the compensated coordinates are expressed in."""
    return 0 if cfg.reference == "first" else t_obs - 1


def compensate_window(window_xy: np.ndarray, video_id: str, start_frame: int,
                      cfg: CMCConfig, t_obs: int | None = None) -> np.ndarray:
    """Compensate a full (T, 2) window into one observed frame's coordinates.

    Used when building training data: observation and target are transformed
    together so they stay in one coordinate system. cfg.reference picks which
    observed frame that is; t_obs is required when it is "last".
    """
    if not cfg.enabled:
        return window_xy
    if cfg.source == "egomotion":
        if cfg.reference == "last" and t_obs is None:
            raise ValueError("t_obs is required when cfg.reference == 'last'")
        ref_idx = 0 if cfg.reference == "first" else reference_index(cfg, t_obs)
        n = len(window_xy)
        H = _ego_homographies(video_id, start_frame, n, n, ref_idx, cfg)
        return apply_homographies(window_xy, H).astype(window_xy.dtype)
    R = rotations_for(video_id, cfg)
    R_win = _slice(R, start_frame, len(window_xy), video_id)
    K = K_for(video_id, cfg)
    if cfg.reference == "first":
        return compensate_trajectory(window_xy, R_win, K).astype(window_xy.dtype)
    if t_obs is None:
        raise ValueError("t_obs is required when cfg.reference == 'last'")
    ref_idx = reference_index(cfg, t_obs)
    return compensate_trajectory(window_xy, R_win, K,
                                 ref_index=ref_idx).astype(window_xy.dtype)


def decompensate_modes(modes: np.ndarray, video_id: str, start_frame: int,
                       t_obs: int, cfg: CMCConfig,
                       true_future: np.ndarray | None = None) -> np.ndarray:
    """Map predicted modes (K, T_pred, 2) back to raw pixels.

    start_frame is the window's first *observed* frame, so the reference
    orientation is R[start_frame] -- the same one compensate_window used.
    """
    if not cfg.enabled:
        return modes
    n_future = modes.shape[-2]
    if cfg.source == "egomotion":
        # frames t_obs .. t_obs+n_future-1 of the window, from observed IMU only
        H = _ego_homographies(video_id, start_frame, t_obs, t_obs + n_future,
                              reference_index(cfg, t_obs), cfg)
        H_back = np.linalg.inv(H[t_obs:])               # reference -> each future frame
        return apply_homographies(modes, H_back)
    R = rotations_for(video_id, cfg)
    R_obs = _slice(R, start_frame, t_obs, video_id)
    R_future = extrapolate(R_obs, n_future, mode=cfg.extrapolation,
                           R_true_future=true_future)
    return inverse_compensate(modes, R_future, K_for(video_id, cfg),
                              R_ref=R_obs[reference_index(cfg, t_obs)])


def true_future_rotations(video_id: str, start_frame: int, t_obs: int,
                          n_future: int, cfg: CMCConfig) -> np.ndarray:
    """The real future rotations, for cfg.extrapolation == 'oracle'."""
    R = rotations_for(video_id, cfg)
    return _slice(R, start_frame + t_obs, n_future, video_id)


# --------------------------------------------------------------------------- #
# Evaluation support
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _tracks():
    from prediction.build_dataset import load_tracks
    return load_tracks()


def raw_future_gt(seq: int, tid: int, start_frame: int, t_obs: int,
                  t_pred: int) -> np.ndarray:
    """Ground truth for the predicted frames, in RAW pixels.

    When CMC is on, the trajectories inside the PPT pickle are compensated, so
    they cannot be used as the target: predictions are mapped back to raw pixels
    and must be scored against raw ground truth, which is what the competition
    measures. This reads it straight from the annotations.
    """
    key = (f"{seq:04d}", tid)
    tracks = _tracks()
    if key not in tracks:
        raise KeyError(f"track {key} not found in annotations")
    frames, xy = tracks[key]
    i = int(np.searchsorted(frames, start_frame))
    if i >= len(frames) or int(frames[i]) != start_frame:
        raise KeyError(f"track {key} has no frame {start_frame}")
    return xy[i + t_obs:i + t_obs + t_pred].astype(np.float64)


def decompensate_batch(modes: np.ndarray, meta: np.ndarray, t_obs: int,
                       cfg: CMCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Map a batch of predictions back to raw pixels and fetch matching GT.

    modes: (N, K, T_pred, 2) absolute compensated pixels
    meta:  (N, 3) rows of (sequence, track_id, first_observed_frame)
    Returns (raw_modes, raw_gt) with shapes (N, K, T_pred, 2) and (N, T_pred, 2).
    """
    n_future = modes.shape[-2]
    raw_modes = np.empty_like(modes, dtype=np.float64)
    raw_gt = np.empty((len(modes), n_future, 2), dtype=np.float64)
    for i, (seq, tid, start) in enumerate(meta):
        seq_id = f"{int(seq):04d}"
        true_future = None
        if cfg.extrapolation == "oracle" and cfg.source == "rotations":
            true_future = true_future_rotations(seq_id, int(start), t_obs,
                                                n_future, cfg)
        raw_modes[i] = decompensate_modes(modes[i], seq_id, int(start), t_obs,
                                          cfg, true_future=true_future)
        raw_gt[i] = raw_future_gt(int(seq), int(tid), int(start), t_obs, n_future)
    return raw_modes, raw_gt


def compensate_observation_only(window_xy: np.ndarray, video_id: str,
                                start_frame: int, t_obs: int,
                                cfg: CMCConfig) -> np.ndarray:
    """Compensate only the observed frames; leave the target in raw pixels.

    The 'input_only' arm. The model sees a de-rotated history but predicts in
    raw pixel space, so no future rotation has to be forecast and none of the
    extrapolation error applies. The cost is that the model must absorb whatever
    the camera does during the prediction window implicitly.

    Reference is the **last** observed frame, not the first. At the reference the
    homography is exactly the identity, so the anchor keeps its true pixel
    position and the raw future stays consistent with it. Referencing the first
    observed frame instead would move the anchor into frame-0 coordinates while
    the target stayed raw, leaving a per-window offset for the model to guess.
    """
    if not cfg.enabled:
        return window_xy
    if cfg.source == "egomotion":
        H = _ego_homographies(video_id, start_frame, t_obs, t_obs, t_obs - 1, cfg)
        out = window_xy.copy()
        out[:t_obs] = apply_homographies(window_xy[:t_obs], H).astype(window_xy.dtype)
        return out
    R = rotations_for(video_id, cfg)
    R_obs = _slice(R, start_frame, t_obs, video_id)
    # Re-reference to the last observed frame by inverting the first-frame
    # convention: R_ref_last = R_obs[-1] means H is identity at t = t_obs-1.
    K = K_for(video_id, cfg)
    H = K @ R_obs[-1] @ np.linalg.inv(R_obs) @ np.linalg.inv(K)
    out = window_xy.copy()
    obs = apply_homographies(window_xy[:t_obs], H)
    out[:t_obs] = obs.astype(window_xy.dtype)
    return out
