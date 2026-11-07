"""Utilities for loading MOT-format tracks and extracting sliding windows."""

from pathlib import Path
from typing import Any

import numpy as np


def load_mot_tracks(
    mot_file: Path | str,
    return_frame_ids: bool = False,
) -> dict[int, Any]:
    """Load MOT-format file into per-track center-point trajectories.

    MOT format: frame_id, track_id, x, y, w, h, conf, class, visibility

    Args:
        mot_file: Path to MOT-format text file.
        return_frame_ids: If True, return dict with 'positions' and 'frame_ids'.

    Returns:
        Dict mapping track_id -> np.ndarray of shape (N, 2) with (cx, cy),
        or dict with 'positions' and 'frame_ids' if return_frame_ids=True.
    """
    mot_file = Path(mot_file)
    raw: dict[int, list[tuple[int, float, float]]] = {}

    text = mot_file.read_text().strip()
    if not text:
        return {}

    for line in text.split("\n"):
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame_id = int(parts[0])
        track_id = int(float(parts[1]))
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        cx = x + w / 2
        cy = y + h / 2
        raw.setdefault(track_id, []).append((frame_id, cx, cy))

    for tid in raw:
        raw[tid].sort(key=lambda t: t[0])

    if return_frame_ids:
        result = {}
        for tid, entries in raw.items():
            result[tid] = {
                "positions": np.array([(e[1], e[2]) for e in entries]),
                "frame_ids": [e[0] for e in entries],
            }
        return result

    return {
        tid: np.array([(e[1], e[2]) for e in entries])
        for tid, entries in raw.items()
    }


def extract_windows(
    positions: np.ndarray,
    frame_ids: list[int],
    t_obs: int,
    t_pred: int,
    stride: int = 1,
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Extract sliding windows of (observation, prediction) pairs.

    Only yields windows where frame_ids are consecutive (no gaps).

    Args:
        positions: (N, 2) array of (cx, cy).
        frame_ids: List of frame IDs corresponding to positions.
        t_obs: Number of observation frames.
        t_pred: Number of prediction frames.
        stride: Sliding window stride.

    Returns:
        List of (obs_positions, pred_positions, metadata) tuples.
    """
    total = t_obs + t_pred
    if len(positions) < total:
        return []

    windows = []
    for start in range(0, len(positions) - total + 1, stride):
        end = start + total
        window_fids = frame_ids[start:end]
        if not _is_consecutive(window_fids):
            continue
        obs = positions[start : start + t_obs]
        pred = positions[start + t_obs : end]
        meta = {
            "start_frame": window_fids[0],
            "obs_end_frame": window_fids[t_obs - 1],
            "pred_end_frame": window_fids[-1],
        }
        windows.append((obs, pred, meta))

    return windows


def _is_consecutive(frame_ids: list[int]) -> bool:
    for i in range(1, len(frame_ids)):
        if frame_ids[i] != frame_ids[i - 1] + 1:
            return False
    return True


def save_predictions_mot(
    predictions: dict[int, dict[int, tuple[float, float]]],
    output_file: Path | str,
) -> None:
    """Save predicted positions in MOT-like format.

    Args:
        predictions: Dict mapping track_id -> {frame_id: (cx, cy)}.
        output_file: Output file path.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for tid in sorted(predictions.keys()):
        for fid in sorted(predictions[tid].keys()):
            cx, cy = predictions[tid][fid]
            lines.append(f"{fid},{tid},{cx:.4f},{cy:.4f}")

    output_file.write_text("\n".join(lines) + "\n")
