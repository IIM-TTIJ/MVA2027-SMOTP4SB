"""Camera orientation per frame: loading, conversion, and synthetic generation.

Convention throughout: `R[t]` is the **world-to-camera** rotation at frame t, so
a point X in world coordinates is R[t] @ X in the camera frame at that instant.
This is what the homography in docs/cmc.md assumes.

Storage format (one file per video):

    {rotation_dir}/{video_id}.npz   with array `R` of shape (T, 3, 3), float64

T is the frame count at `fps` (30 by default), so R[t] lines up with video frame
t. Real IMU data will arrive at a different rate and time base; converting it
into this format is the loader's job, and everything downstream is unchanged.

Usage:
    python -m cmc.rotations --make-synthetic 0001 --frames 300
"""

import argparse
import csv

import numpy as np
from scipy.spatial.transform import Rotation

from cmc.config import CMCConfig
from prediction import paths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def imu_path(video_id: str, split: str):
    """The sequence's IMU log, or None. Only siyi_a8mini sequences ship one."""
    p = paths.SMOT4SB / split / str(video_id) / "imu.csv"
    return p if p.is_file() else None


def has_rotation_source(video_id: str, cfg: CMCConfig) -> bool:
    """Whether load_rotations would succeed: a cached .npz or an imu.csv."""
    return ((cfg.rotation_dir / f"{video_id}.npz").exists()
            or imu_path(video_id, cfg.split) is not None)


# Camera axes (x right, y down, z optical axis) expressed in the drone's body
# axes (x forward, y right, z down, as ArduPilot logs them): the camera looks
# along the body's forward axis, its x is the body's right, its y the body's down.
CAMERA_IN_BODY = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    flip = np.linalg.det(out) < 0                  # a reflection is not a rotation
    if flip.any():
        U[flip, :, -1] *= -1
        out = U @ Vt
    return out


def load_imu_rotations(path) -> np.ndarray:
    """(T, 3, 3) world-to-camera rotations from the dataset's imu.csv.

    imu.csv carries the drone **body** attitude: R00..R22 is the body-to-NED
    rotation Rz(yaw) Ry(pitch) Rx(roll) (SMOT4SB/tools/extract_imu.py), not a
    camera rotation. The camera hangs on a 3-axis gimbal that follows the
    drone's yaw and holds pitch and roll level, so:

      gimbal_retracted = 0   camera-to-NED = Rz(yaw) @ CAMERA_IN_BODY
                             (only the heading reaches the image)
      gimbal_retracted = 1   camera-to-NED = R_body @ CAMERA_IN_BODY
                             (the camera is fixed to the airframe)

    and the returned world-to-camera rotation is the transpose. The gimbal's
    own tilt is not logged; it is taken as level, and a constant tilt does not
    change the frame-to-frame rotation for a pure heading change to first order.

    The CSV holds six decimals, so each body matrix is projected back onto the
    rotation group (SVD) first. The frame column must be exactly 1..T, since
    R[t] is taken to be frame t+1 downstream.
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: empty IMU log")
    frames = [int(r["frame"]) for r in rows]
    if frames != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: frame column is not 1..{len(rows)} without gaps")
    R_body = _project_to_so3(np.array(
        [[[float(r[f"R{i}{j}"]) for j in range(3)] for i in range(3)] for r in rows],
        dtype=np.float64))
    retracted = np.array([int(r.get("gimbal_retracted", 0) or 0) for r in rows], dtype=bool)

    yaw = np.arctan2(R_body[:, 1, 0], R_body[:, 0, 0])      # heading, from the matrix itself
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.zeros_like(R_body)
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
    R_nav_cam = np.where(retracted[:, None, None], R_body, Rz) @ CAMERA_IN_BODY
    return np.transpose(R_nav_cam, (0, 2, 1))


def load_rotations(video_id: str, cfg: CMCConfig | None = None) -> np.ndarray:
    """Return (T, 3, 3) world-to-camera rotations for one video.

    Source precedence: a cached {rotation_dir}/{video_id}.npz, else the
    sequence's own imu.csv. Either way the result must cover the sequence's
    frames exactly (check_covers).
    """
    cfg = cfg or CMCConfig()
    path = cfg.rotation_dir / f"{video_id}.npz"
    imu = imu_path(video_id, cfg.split)
    if path.exists():
        R = np.load(path)["R"].astype(np.float64)
    elif imu is not None:
        R = load_imu_rotations(imu)
    else:
        raise SystemExit(
            f"no rotation source for video {video_id}: no {path} and no imu.csv "
            f"in the sequence directory.\n"
            f"  Only siyi_a8mini sequences carry an IMU log; the rest "
            f"need a visual estimate, which does not exist yet.\n"
            f"  --make-synthetic writes identity placeholders for unit tests "
            f"only -- a run built on them is not a result.")
    R = check_rotations(R)
    n = frame_count(video_id, cfg.split)
    if n is not None:
        check_covers(R, video_id, n)
    return R


def frame_count(video_id: str, split: str) -> int | None:
    """Frames in the sequence's image directory, or None if there is no such
    sequence -- the synthetic ids the tests invent, or no dataset mounted."""
    d = paths.SMOT4SB / split / str(video_id)
    if not d.is_dir():
        return None
    return sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def check_covers(R: np.ndarray, video_id: str, n_frames: int) -> np.ndarray:
    """Refuse a rotation file that does not belong to this footage.

    A file is named for a sequence id, and a cached file can outlive the
    footage it was computed from. Length is the one property that can be
    checked without the IMU log. A file that is too short fails later anyway with a confusing frame-range
    error; one that is too long is read without complaint, which is worse.
    """
    if len(R) != n_frames:
        raise ValueError(
            f"video {video_id}: rotation file has {len(R)} frames but the "
            f"sequence has {n_frames} -- it describes different footage.\n"
            f"  Regenerate it:  python -m cmc.rotations --make-synthetic "
            f"{video_id} --mode identity\n"
            f"  (--frames defaults to the sequence's frame count)")
    return R


def save_rotations(video_id: str, R: np.ndarray, cfg: CMCConfig | None = None):
    cfg = cfg or CMCConfig()
    cfg.rotation_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.rotation_dir / f"{video_id}.npz"
    np.savez_compressed(path, R=check_rotations(R))
    return path


def check_rotations(R: np.ndarray) -> np.ndarray:
    """Reject anything that is not a stack of proper rotation matrices.

    A silently non-orthonormal R produces a homography that quietly shears the
    trajectory, which is far harder to spot later than an exception here.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"rotations must be (T, 3, 3), got {R.shape}")
    should_be_I = R @ np.transpose(R, (0, 2, 1))
    if not np.allclose(should_be_I, np.eye(3), atol=1e-6):
        worst = np.abs(should_be_I - np.eye(3)).max()
        raise ValueError(f"rotations are not orthonormal (max |RR^T - I| = {worst:.2e})")
    dets = np.linalg.det(R)
    if not np.allclose(dets, 1.0, atol=1e-6):
        raise ValueError(f"rotations must have det=+1, got range "
                         f"[{dets.min():.6f}, {dets.max():.6f}] (reflection?)")
    return R


# --------------------------------------------------------------------------- #
# Converters -- the real IMU format is still unknown, so support the likely ones
# --------------------------------------------------------------------------- #
def from_quaternion(q: np.ndarray, scalar_first: bool = False) -> np.ndarray:
    """(T, 4) quaternions -> (T, 3, 3) rotations.

    scipy expects scalar-LAST (x, y, z, w). Many flight logs, including DJI's,
    write scalar-FIRST (w, x, y, z). Passing the wrong one yields a valid-looking
    rotation that is simply wrong, so the caller must state which it has.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quaternions must be (T, 4), got {q.shape}")
    if scalar_first:
        q = q[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("quaternion array contains a zero-norm entry")
    return check_rotations(Rotation.from_quat(q / norms[:, None]).as_matrix())


def from_euler(angles: np.ndarray, seq: str = "ZYX", degrees: bool = True) -> np.ndarray:
    """(T, 3) Euler angles -> (T, 3, 3) rotations.

    `seq` defaults to yaw-pitch-roll, the usual aerospace order. Euler order is
    not self-describing in a log file; confirm it against the source.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(f"euler angles must be (T, 3), got {angles.shape}")
    return check_rotations(Rotation.from_euler(seq, angles, degrees=degrees).as_matrix())


# --------------------------------------------------------------------------- #
# Synthetic generation
# --------------------------------------------------------------------------- #
def synthetic(n_frames: int, mode: str = "sinusoidal", fps: float = 30.0,
              pan_amp_deg: float = 6.0, pan_hz: float = 0.15,
              tilt_amp_deg: float = 2.5, tilt_hz: float = 0.23,
              pan_rate_deg: float = 3.0, jitter_amp_deg: float = 0.5,
              jitter_hz: float = 1.7, seed: int = 0) -> np.ndarray:
    """Generate (T, 3, 3) camera rotations for testing without real IMU data.

    mode='identity'   -- every frame is the identity. Used by the identity test
                         to prove that enabling CMC with a still camera changes
                         nothing at all.
    mode='sinusoidal' -- pan and tilt angular velocities are independent sinusoids
                         which are then integrated into orientations. Two
                         incommensurate frequencies keep the motion from being
                         exactly periodic over the window, which would let a
                         predictor learn the camera instead of the bird.

    Amplitudes are angular *velocity* in deg/s; the defaults give a few degrees
    of sway, in the range of hand-held drone footage.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    if mode == "identity":
        return np.tile(np.eye(3), (n_frames, 1, 1))
    if mode not in ("sinusoidal", "pan_jitter"):
        raise ValueError(f"unknown synthetic mode {mode!r}")

    dt = 1.0 / fps
    t = np.arange(n_frames) * dt
    # Angular velocity about the camera's y (pan) and x (tilt) axes, rad/s.
    omega = np.zeros((n_frames, 3))
    if mode == "sinusoidal":
        omega[:, 1] = np.deg2rad(pan_amp_deg) * np.sin(2 * np.pi * pan_hz * t)
        omega[:, 0] = np.deg2rad(tilt_amp_deg) * np.sin(2 * np.pi * tilt_hz * t + 1.0)
    else:
        # A steady pan with small shake on top -- what a pilot tracking a bird
        # actually does. This matters for the comparison: a pure sinusoid
        # reverses direction, which is the worst case for constant-velocity
        # extrapolation, whereas a sustained pan is its best case. Reporting
        # only the sinusoid would understate constant_omega.
        omega[:, 1] = np.deg2rad(pan_rate_deg) + \
            np.deg2rad(jitter_amp_deg) * np.sin(2 * np.pi * jitter_hz * t)
        omega[:, 0] = np.deg2rad(jitter_amp_deg) * 0.6 * \
            np.sin(2 * np.pi * jitter_hz * 1.3 * t + 0.7)

    # Integrate: each frame's increment is applied in the camera's current frame,
    # so increments compose on the left. Building it with from_rotvec keeps every
    # step an exact rotation instead of accumulating small-angle drift.
    steps = Rotation.from_rotvec(omega * dt)
    R = np.empty((n_frames, 3, 3))
    current = Rotation.identity()
    for i in range(n_frames):
        current = steps[i] * current
        R[i] = current.as_matrix()
    return check_rotations(R)


def concatenate(*sequences: np.ndarray) -> np.ndarray:
    """Join rotation sequences so orientation stays continuous across the joins.

    Each later segment is composed onto where the previous one ended, rather
    than restarting from identity -- otherwise the join is a discontinuity that
    shows up as a huge spurious angular velocity spike.

    Used to build piecewise flights (pan phase, then hover/oscillation) for
    testing the per-window classifier.
    """
    if not sequences:
        raise ValueError("nothing to concatenate")
    out = [np.asarray(sequences[0], dtype=np.float64)]
    for nxt in sequences[1:]:
        out.append(np.asarray(nxt, dtype=np.float64) @ out[-1][-1])
    return check_rotations(np.concatenate(out))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate or inspect camera rotations")
    p.add_argument("--make-synthetic", metavar="VIDEO_ID",
                   help="write a synthetic rotation file for this video id")
    p.add_argument("--frames", type=int, default=None,
                   help="rotation count; defaults to the sequence's frame count "
                        "when it is a dataset sequence, else 300")
    p.add_argument("--mode", default="sinusoidal",
                   choices=["sinusoidal", "pan_jitter", "identity"])
    p.add_argument("--inspect", metavar="VIDEO_ID", help="summarise an existing file")
    a = p.parse_args()
    cfg = CMCConfig().validate()

    if a.make_synthetic:
        frames = a.frames or frame_count(a.make_synthetic, cfg.split) or 300
        R = synthetic(frames, mode=a.mode, fps=cfg.fps)
        path = save_rotations(a.make_synthetic, R, cfg)
        print(f"  wrote {len(R)} frames ({a.mode}) -> {path}")
    if a.inspect:
        R = load_rotations(a.inspect, cfg)
        rv = Rotation.from_matrix(R).as_rotvec()
        print(f"  {a.inspect}: {len(R)} frames, "
              f"total rotation {np.rad2deg(np.linalg.norm(rv, axis=1)).max():.2f} deg max")
    if not (a.make_synthetic or a.inspect):
        p.error("nothing to do: pass --make-synthetic or --inspect")


if __name__ == "__main__":
    main()
