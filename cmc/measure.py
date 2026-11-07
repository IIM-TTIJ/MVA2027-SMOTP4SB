"""Camera motion measured on the frames themselves, used to fit cmc.egomotion.

For each consecutive pair: sparse Lucas-Kanade tracks on a 480-pixel-wide
grayscale copy, a forward-backward consistency check, then a RANSAC similarity
fit (cv2.estimateAffinePartial2D) so that the dominant, static background wins
over moving water or birds. Output per transition: dx, dy (pixels at the
sequence's own resolution), ln(scale), rotation (rad), inlier count; NaN where
the fit fails.

This is training-time supervision only. Compensation at run time uses the IMU.
"""

import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

WORK_W = 480
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load(path: str) -> tuple[np.ndarray, int]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    h, w = img.shape
    small = cv2.resize(img, (WORK_W, round(h * WORK_W / w)), interpolation=cv2.INTER_AREA)
    return small, w


def pair(a: np.ndarray, b: np.ndarray, scale: float) -> list[float]:
    """Similarity motion from a to b, in full-resolution pixels."""
    nan = [np.nan] * 4 + [0.0]
    pts = cv2.goodFeaturesToTrack(a, maxCorners=800, qualityLevel=0.005, minDistance=8, blockSize=7)
    if pts is None or len(pts) < 12:
        return nan
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, winSize=(21, 21), maxLevel=3)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
    if ok.sum() < 12:
        return nan
    c = np.array([a.shape[1] / 2.0, a.shape[0] / 2.0], dtype=np.float32)
    p0 = pts.reshape(-1, 2)[ok] - c
    p1 = nxt.reshape(-1, 2)[ok] - c
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                         maxIters=2000, confidence=0.995)
    if M is None:
        return nan
    s = float(np.hypot(M[0, 0], M[1, 0]))
    return [float(M[0, 2]) * scale, float(M[1, 2]) * scale, float(np.log(s)),
            float(np.arctan2(M[1, 0], M[0, 0])), float(inl.sum())]


def _chunk(paths: list[str]) -> np.ndarray:
    cv2.setNumThreads(1)
    prev, w = _load(paths[0])
    out = []
    for p in paths[1:]:
        cur, _ = _load(p)
        out.append(pair(prev, cur, w / WORK_W))
        prev = cur
    return np.array(out, dtype=np.float64).reshape(-1, 5)


def frame_paths(seq_dir) -> list[str]:
    d = Path(seq_dir)
    return [str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]


def measure_sequence(seq_dir, workers: int = 8, step: int = 100) -> np.ndarray:
    """(T-1, 5) per-transition motion of the frames in seq_dir (sorted by name)."""
    paths = frame_paths(seq_dir)
    if len(paths) < 2:
        return np.zeros((0, 5))
    chunks = [paths[i:i + step + 1] for i in range(0, len(paths) - 1, step)]
    if workers <= 1:
        res = [_chunk(c) for c in chunks]
    else:
        with Pool(min(workers, len(chunks), os.cpu_count() or 1)) as pool:
            res = pool.map(_chunk, chunks)
    return np.vstack(res)


# --------------------------------------------------------------------------- #
# Shared by fitting and evaluation
# --------------------------------------------------------------------------- #
REF_WIDTH = 1920
# Per-transition limits at REF_WIDTH; beyond them the measurement, not the
# camera, is what went wrong (blur at full zoom, featureless sky).
MAX_SHIFT = 300.0
MAX_LOG_S = 0.3


def cached_motion(seq_dir, cache, key: str, workers: int = 8) -> np.ndarray:
    """measure_sequence, stored as <cache>/<key with / -> _>.npy."""
    if cache is not None:
        f = Path(cache) / f"{key.replace('/', '_')}.npy"
        if f.exists():
            return np.load(f)
    m = measure_sequence(seq_dir, workers=workers)
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache) / f"{key.replace('/', '_')}.npy", m)
    return m


def clean_motion(motion: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(dx, dy, ln s) with failed or implausible transitions zeroed, and the bad mask."""
    Y = motion[:, :3].copy()
    lim = MAX_SHIFT * width / REF_WIDTH
    bad = (~np.isfinite(Y).all(1)) | (np.abs(Y[:, 0]) > lim) | (np.abs(Y[:, 1]) > lim) \
        | (np.abs(Y[:, 2]) > MAX_LOG_S)
    Y[bad] = 0.0
    return Y, bad
