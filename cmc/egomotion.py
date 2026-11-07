"""Camera motion from the IMU: per-frame image motion predicted from imu.csv.

Why not a rotation homography. The ProDrone camera is not fixed to the
airframe, and the IMU does not see it directly:

  * the gimbal follows the drone's heading, but smoothed, roughly 0.5-1 s late;
  * the pilot pans and tilts the gimbal independently (RC12 / RC16, rate
    inputs logged as gimbal_yaw_cmd / gimbal_pitch_cmd);
  * the lens zooms optically (RC15, logged as zoom_cmd), by up to ~10x within
    a sequence, so the focal length is not a constant.

Neither the gimbal angles nor the zoom ratio are logged, so K R K^-1 cannot be
formed. Instead a linear filter predicts, for every frame transition t -> t+1,
the image motion as a similarity about the image centre:

    x_{t+1} = s_t (x_t - c) + c + (dx_t, dy_t)

from the IMU channels around t (lagged so that the follow delay, the gimbal's
response and small timing offsets are learned rather than modelled). It is a
ridge regression fitted on training sequences against motion measured on their
own frames (cmc.measure, tools/fit_egomotion.py).

At run time it reads only the imu.csv rows it is given -- the frames of the
window being compensated -- and pads the lags with zeros at the edges, exactly
as in training. Track 2 passes the observed frames alone, so no future IMU is
used.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channels and the lags (in frames) each is read at, relative to transition t.
# body: the drone's own yaw step; the gimbal follows it late, hence the long
# history. rpitch: pitch step while the gimbal is retracted (camera fixed to
# the airframe).
LAGS: dict[str, tuple[int, int]] = {
    "gy": (-8, 8),
    "gp": (-8, 8),
    "zc": (-8, 8),
    "body": (-45, 5),
    "rpitch": (-3, 3),
}
REQUIRED = ("frame", "yaw_deg", "pitch_deg", "gimbal_retracted",
            "gimbal_yaw_cmd", "gimbal_pitch_cmd", "zoom_cmd")
OUTPUTS = ("dx", "dy", "log_s")


# --------------------------------------------------------------------------- #
# imu.csv
# --------------------------------------------------------------------------- #
def load_imu(path) -> dict[str, np.ndarray]:
    """Columns of a sequence's imu.csv as float arrays, one row per frame.

    Rejects the pre-2026-10 format (body attitude R00..R22 without gimbal
    commands): those files were attached to footage the logs do not belong to.
    """
    path = Path(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            if "R00" in header:
                raise ValueError(
                    f"{path} is an old-format imu.csv (body attitude only). Those files were "
                    "matched to the wrong footage and are withdrawn; use the rebuilt imu.csv "
                    "(columns gimbal_yaw_cmd, gimbal_pitch_cmd, zoom_cmd, ...).")
            raise ValueError(f"{path} lacks columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {}
    for c in header:
        try:
            out[c] = np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
        except ValueError:
            continue                                   # text columns (rec, log)
    frames = out["frame"]
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(f"{path}: frame column must run 1..{len(frames)} without gaps")
    return out


def imu_window(imu: dict[str, np.ndarray], start_frame: int, n: int) -> dict[str, np.ndarray]:
    """Rows of frames start_frame .. start_frame+n-1 (1-based, as in the annotations)."""
    i = start_frame - 1
    T = len(imu["frame"])
    if i < 0 or i + n > T:
        raise ValueError(f"need imu rows for frames {start_frame}..{start_frame + n - 1}, have 1..{T}")
    return {k: v[i:i + n] for k, v in imu.items()}


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def channels(imu: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-transition channels (length n-1) from n rows."""
    yaw = np.unwrap(np.radians(imu["yaw_deg"]))
    pitch = np.radians(imu["pitch_deg"])
    ret = np.nan_to_num(imu["gimbal_retracted"][:-1])
    return {
        "gy": np.nan_to_num(imu["gimbal_yaw_cmd"][:-1]),
        "gp": np.nan_to_num(imu["gimbal_pitch_cmd"][:-1]),
        "zc": np.nan_to_num(imu["zoom_cmd"][:-1]),
        "body": np.nan_to_num(-np.diff(yaw)),
        "rpitch": np.nan_to_num(ret * np.diff(pitch)),
    }


def feature_names() -> list[str]:
    return [f"{ch}[{l:+d}]" for ch, (lo, hi) in LAGS.items() for l in range(lo, hi + 1)]


def features(imu: dict[str, np.ndarray]) -> np.ndarray:
    """(n-1, F) lagged channels; lags that fall outside the given rows are zero."""
    ch = channels(imu)
    m = len(ch["gy"])
    cols = []
    for name, (lo, hi) in LAGS.items():
        v = ch[name]
        for l in range(lo, hi + 1):
            c = np.zeros(m)
            if l >= 0:
                c[:max(m - l, 0)] = v[l:]
            else:
                c[min(-l, m):] = v[:max(m + l, 0)]
            cols.append(c)
    return np.stack(cols, 1) if m else np.zeros((0, len(feature_names())))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class EgoMotionModel:
    """Ridge regression from features() to (dx, dy, ln s) per transition.

    dx, dy are in pixels of an image of size `image_size`; predictions are
    rescaled to the sequence's resolution by predict(width=...).
    """
    mean: np.ndarray
    scale: np.ndarray
    W: np.ndarray            # (F, 3)
    bias: np.ndarray         # (3,)
    image_size: tuple[int, int] = (1920, 1080)
    lam: float = 3000.0
    # Shrinks every prediction. Compensation errors accumulate over a window, so
    # over-compensating costs more than under-compensating; leave-one-recording-
    # out validation on the training sequences put the best value near 0.5.
    gain: float = 1.0
    trained_on: tuple[str, ...] = ()

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, weight: np.ndarray | None = None, lam: float = 3000.0,
            gain: float = 1.0, image_size=(1920, 1080), trained_on=()) -> "EgoMotionModel":
        if X.shape[1] != len(feature_names()):
            raise ValueError(f"X has {X.shape[1]} features, expected {len(feature_names())}")
        w = np.ones(len(X)) if weight is None else np.asarray(weight, float)
        sw = w.sum()
        if sw <= 0:
            raise ValueError("no training transitions with positive weight")
        mean = (X * w[:, None]).sum(0) / sw
        scale = np.sqrt(((X - mean) ** 2 * w[:, None]).sum(0) / sw) + 1e-9
        Xs = (X - mean) / scale
        ybar = (Y * w[:, None]).sum(0) / sw
        A = Xs.T @ (Xs * w[:, None]) + lam * np.eye(X.shape[1])
        W = np.linalg.solve(A, Xs.T @ ((Y - ybar) * w[:, None]))
        return cls(mean=mean, scale=scale, W=W, bias=ybar, image_size=tuple(image_size), lam=lam,
                   gain=gain, trained_on=tuple(trained_on))

    def predict(self, X: np.ndarray, width: int | None = None) -> np.ndarray:
        P = self.gain * (((X - self.mean) / self.scale) @ self.W + self.bias)
        if width is not None and width != self.image_size[0]:
            P = P.copy()
            P[:, :2] *= width / self.image_size[0]
        return P

    def predict_imu(self, imu: dict[str, np.ndarray], width: int | None = None) -> np.ndarray:
        return self.predict(features(imu), width)

    # JSON, so that the shipped model is readable and diffable
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "format": "smotp4sb-egomotion/1", "lags": LAGS, "outputs": OUTPUTS,
            "features": feature_names(), "image_size": list(self.image_size), "lam": self.lam, "gain": self.gain,
            "trained_on": list(self.trained_on), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "W": self.W.tolist(), "bias": self.bias.tolist()}, indent=1))

    @classmethod
    def load(cls, path) -> "EgoMotionModel":
        raw = json.loads(Path(path).read_text())
        if raw.get("format") != "smotp4sb-egomotion/1":
            raise ValueError(f"{path} is not an egomotion model file")
        if raw["features"] != feature_names():
            raise ValueError(f"{path} was fitted with different feature lags; refit it "
                             "(tools/fit_egomotion.py)")
        return cls(mean=np.array(raw["mean"]), scale=np.array(raw["scale"]), W=np.array(raw["W"]),
                   bias=np.array(raw["bias"]), image_size=tuple(raw["image_size"]), lam=raw["lam"],
                   gain=float(raw.get("gain", 1.0)), trained_on=tuple(raw["trained_on"]))


# --------------------------------------------------------------------------- #
# Motion -> homographies
# --------------------------------------------------------------------------- #
def similarity(dx: float, dy: float, log_s: float, cx: float, cy: float) -> np.ndarray:
    """3x3 matrix of x' = s (x - c) + c + d."""
    s = float(np.exp(log_s))
    return np.array([[s, 0.0, (1 - s) * cx + dx],
                     [0.0, s, (1 - s) * cy + dy],
                     [0.0, 0.0, 1.0]])


def homographies_to_ref(motion: np.ndarray, ref_index: int, center: tuple[float, float]) -> np.ndarray:
    """(n, 3, 3) maps from each frame's pixels to frame ref_index's.

    motion: (n-1, 3) per-transition (dx, dy, ln s). A_t maps frame 0 to frame
    t (A_0 = I, A_{t+1} = S_t A_t), so frame t -> ref is A_ref A_t^-1.
    """
    n = len(motion) + 1
    if not 0 <= ref_index < n:
        raise ValueError(f"ref_index {ref_index} outside 0..{n - 1}")
    cx, cy = center
    A = np.empty((n, 3, 3))
    A[0] = np.eye(3)
    for t, (dx, dy, ls) in enumerate(motion):
        A[t + 1] = similarity(dx, dy, ls, cx, cy) @ A[t]
    return A[ref_index] @ np.linalg.inv(A)
