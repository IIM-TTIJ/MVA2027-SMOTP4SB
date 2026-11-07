"""Per-sequence camera intrinsics, read from the dataset's camera_intrinsics.json.

K is a property of the sequence, not of the run. SMOT4SB was filmed with three
drone cameras at two resolutions, and the focal lengths differ by a factor of
two across them:

    mavic2pro    1920x1080   f = 1492
    mavic2pro    3840x2160   f = 2985
    phantom4pro  3840x2160   f = 2560
    siyi_a8mini  3840x2160   f = 2248

A single run-wide K is therefore wrong for most of the data, which matters
because the compensation homography is K R K^-1: an error in f scales the
de-rotation, and an error in the principal point shifts its centre.

Entries are keyed "{split}/{sequence}" because sequence ids repeat across
splits -- train/0001 and pub_test/0001 are different footage -- so a bare
sequence id cannot identify a camera on its own. CMCConfig.split says which
split ids are resolved against.
"""

import functools
import json
from pathlib import Path

import numpy as np


@functools.lru_cache(maxsize=8)
def _sequences(path: str) -> dict:
    """The 'sequences' block of camera_intrinsics.json, read once per file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"camera intrinsics not found at {p}\n"
            "  camera_intrinsics.json ships with the dataset; set SMOT4SB_ROOT "
            "so the default resolves, or point CMCConfig.intrinsics at it.")
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("sequences"), dict):
        raise ValueError(f"{p} has no 'sequences' object")
    return raw["sequences"]


@functools.lru_cache(maxsize=512)
def _K_cached(key: str, path: str) -> np.ndarray:
    """One parse and validation per sequence, not per window.

    There are tens of thousands of windows and hundreds of sequences, so this is the
    same reason cmc.pipeline caches rotations by video id.
    """
    sequences = _sequences(path)
    entry = sequences.get(key)
    if entry is None:
        split = key.split("/")[0]
        known = sorted(k for k in sequences if k.startswith(f"{split}/"))
        raise KeyError(
            f"no camera intrinsics for {key!r} in {path}\n"
            f"  {len(known)} sequences are listed for split {split!r}"
            + (f" ({known[0]} .. {known[-1]})" if known else "")
            + "\n  Check CMCConfig.split: ids repeat across splits, so the "
              "split has to match the annotations the windows came from.")

    if "K" not in entry:
        raise ValueError(f"{key}: entry has no 'K'")
    try:
        K = np.asarray(entry["K"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: K is not numeric ({exc})") from None
    if K.shape != (3, 3):
        raise ValueError(f"{key}: K must be 3x3, got {K.shape}")
    if not np.isfinite(K).all():
        raise ValueError(f"{key}: K contains non-finite values")
    if K[2, 2] == 0:
        raise ValueError(f"{key}: K[2,2] is zero; intrinsics are not normalised")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{key}: focal lengths must be positive, got "
                         f"fx={K[0, 0]}, fy={K[1, 1]}")

    # The principal point should sit at the centre of the frame it was derived
    # for. This holds for every sequence, and it is the check that would have
    # caught the old placeholder K, whose centre was correct for 4K but off by a
    # factor of two on the sequences shot at 1920x1080.
    resolution = entry.get("resolution")
    if resolution:
        w, h = float(resolution[0]), float(resolution[1])
        if not (np.isclose(K[0, 2], w / 2) and np.isclose(K[1, 2], h / 2)):
            raise ValueError(
                f"{key}: principal point ({K[0, 2]}, {K[1, 2]}) is not the "
                f"centre of the stated {int(w)}x{int(h)} frame")

    K.flags.writeable = False        # it is cached and shared; nobody may edit it
    return K


def K_for(video_id, cfg) -> np.ndarray:
    """The (3, 3) intrinsic matrix for one sequence, as float64.

    video_id is a bare sequence id such as "0086"; cfg.split qualifies it.
    The returned array is cached and read-only -- copy it before mutating.
    """
    return _K_cached(f"{cfg.split}/{video_id}", str(cfg.intrinsics))
