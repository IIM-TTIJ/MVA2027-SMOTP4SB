"""SO-HOTA-TP: SO-HOTA for Trajectory Prediction.

Adapts SO-HOTA (Small Object HOTA) from MVA2025 SMOT4SB to evaluate
trajectory prediction under frequent object disappearance.

SO-HOTA-TP = sqrt(SO-DetA-TP * SO-AssA-TP), averaged across alpha thresholds.
Similarity is DotDistance = exp(-d/S) where d is Euclidean centroid
distance and S = sqrt(mean GT bounding box area).

Key differences from standard ADE/FDE:
  - Handles disappearing objects via FP/FN counting
  - Evaluates identity consistency (AssA) across prediction horizon
  - Multi-threshold evaluation (alpha 0.05-0.95) captures both
    coarse and fine-grained prediction quality

References:
  - HOTA: Luiten et al., IJCV 2021
  - SO-HOTA: MVA2025 SMOT4SB Challenge
  - Definition: docs/so_hota_tp.md

Usage:
    python -m evaluation.so_hota_tp \\
        --pred pred.txt --gt gt.txt \\
        [--tracks tracker_output.txt]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

T_OBS_DEFAULT = 60
T_PRED_DEFAULT = 150
FIRST_FRAME = 1        # images are 00001.jpg; windows are cut from here
ALPHA_RANGE = (0.05, 0.99, 0.05)


def association_max_distance(S: float, alpha_min: float = ALPHA_RANGE[0]) -> float:
    """Largest centre distance that can still count as a match, in pixels.

    Submitted track ids are arbitrary, so they have to be associated with GT
    ids before scoring. That association needs a single cutoff, whereas the
    metric itself has none -- it sweeps alpha over DotD space.

    Derive the cutoff from the weakest alpha the metric uses. Inverting
    DotD = exp(-d/S) >= alpha at alpha_min gives the distance beyond which no
    alpha in the sweep would ever call the pair a true positive, so associating
    anything further apart admits a pair the metric cannot score. Because it is
    a function of S, it rescales with the dataset instead of being a magic
    constant.

    At S = 24.15 px (SMOT4SB) and alpha_min = 0.05 this is 72.3 px, about 3x the
    mean bounding box.
    """
    return -S * float(np.log(alpha_min))


def compute_S_from_gt(gt_file: Path | str) -> float:
    """Compute S = sqrt(mean GT bounding box area) from a MOT-format file.

    This is the dataset-wide normalization factor for DotDistance, shared
    between Track 1 (SO-HOTA) and Track 2 (SO-HOTA-TP).

    Args:
        gt_file: Path to MOT-format ground truth file.

    Returns:
        S value (scalar).
    """
    areas: list[float] = []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            w, h = float(parts[4]), float(parts[5])
            if w > 0 and h > 0:
                areas.append(w * h)
    if not areas:
        raise ValueError(f"No valid bounding boxes found in {gt_file}")
    return float(np.sqrt(np.mean(areas)))


def dot_distance_matrix(
    gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    S: float,
) -> np.ndarray:
    """Compute pairwise DotDistance matrix.

    DotDistance(A, B) = exp(-d(A, B) / S)

    Args:
        gt_pos: (N_gt, 2) GT centroid positions.
        pred_pos: (N_pred, 2) predicted centroid positions.
        S: Normalization factor.

    Returns:
        (N_gt, N_pred) DotDistance matrix in [0, 1].
    """
    if len(gt_pos) == 0 or len(pred_pos) == 0:
        return np.zeros((len(gt_pos), len(pred_pos)))
    dist = np.linalg.norm(
        gt_pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :], axis=2
    )
    return np.exp(-dist / S)


def _load_gt_tracks_with_bbox(
    gt_file: Path | str,
) -> dict[int, dict[str, Any]]:
    """Load GT MOT file into per-track data with frame IDs and positions.

    Returns:
        Dict mapping track_id -> {
            'frame_ids': list[int],
            'positions': np.ndarray of shape (N, 2) with (cx, cy),
            'bboxes': np.ndarray of shape (N, 4) with (x, y, w, h),
        }
    """
    raw: dict[int, list[tuple[int, float, float, float, float, float, float]]] = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(float(parts[1]))
            x, y, w, h = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            cx, cy = x + w / 2, y + h / 2
            raw.setdefault(tid, []).append((fid, cx, cy, x, y, w, h))

    result: dict[int, dict[str, Any]] = {}
    for tid, entries in raw.items():
        entries.sort(key=lambda t: t[0])
        result[tid] = {
            "frame_ids": [e[0] for e in entries],
            "positions": np.array([(e[1], e[2]) for e in entries]),
            "bboxes": np.array([(e[3], e[4], e[5], e[6]) for e in entries]),
        }
    return result


def _load_predictions(
    pred_file: Path | str,
) -> dict[int, dict[int, dict[int, dict[int, np.ndarray]]]]:
    """Load a prediction file, keyed by the window it was made from.

    Submission format (6 columns):

        window_start, frame_id, track_id, mode_k, cx, cy

    `window_start` is the first observed frame of the window the prediction was
    made from, and it is what makes a row unique. Without it a sequence's
    windows collide: windows advance by t_pred, so one frame is inside the
    prediction horizon of more than one window, and the same
    (frame_id, track_id, mode_k) then names two different predictions -- made
    from different observations, and both legitimate. Keying only on those three
    silently keeps whichever row was read last.

    The legacy 4/5-column form (frame_id, track_id, cx, cy[, mode_k]) is still
    read, and lands under window_start = -1. It cannot express more than one
    window, so use it only for single-window fixtures.

    Returns:
        window_start -> track_id -> mode_k -> frame_id -> np.array([cx, cy])
    """
    pred: dict[int, dict[int, dict[int, dict[int, np.ndarray]]]] = {}
    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                win = int(float(parts[0]))
                fid = int(float(parts[1]))
                tid = int(float(parts[2]))
                k = int(float(parts[3]))
                cx, cy = float(parts[4]), float(parts[5])
            elif len(parts) >= 4:
                win = -1
                fid = int(float(parts[0]))
                tid = int(float(parts[1]))
                cx, cy = float(parts[2]), float(parts[3])
                k = int(float(parts[4])) if len(parts) > 4 else 0
            else:
                continue
            pred.setdefault(win, {}).setdefault(tid, {}).setdefault(k, {})[fid] = (
                np.array([cx, cy])
            )
    return pred


def select_best_mode(
    pred_modes: dict[int, dict[int, np.ndarray]],
    gt_frame_positions: dict[int, np.ndarray],
    S: float,
) -> dict[int, np.ndarray]:
    """Select best-of-K mode for one track via average DotDistance.

    Args:
        pred_modes: mode_k -> {frame_id: position}
        gt_frame_positions: {frame_id: position} for GT-visible frames
        S: DotDistance normalization factor.

    Returns:
        Selected mode: {frame_id: position}
    """
    if not pred_modes:
        return {}

    best_k = 0
    best_score = -1.0

    for k, mode_frames in pred_modes.items():
        scores: list[float] = []
        for fid, gt_pos in gt_frame_positions.items():
            if fid in mode_frames:
                d = float(np.linalg.norm(mode_frames[fid] - gt_pos))
                scores.append(np.exp(-d / S))
        avg_score = float(np.mean(scores)) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_k = k

    return pred_modes[best_k]


def _build_window_frame_data(
    gt_tracks: dict[int, dict[str, Any]],
    pred_selected: dict[int, dict[int, np.ndarray]],
    pred_start: int,
    t_pred: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build per-frame GT and prediction data for one evaluation window.

    Returns:
        (gt_ids_per_frame, gt_pos_per_frame,
         pred_ids_per_frame, pred_pos_per_frame)
        Each is a list of length t_pred. gt_ids/pred_ids are 1D int arrays,
        gt_pos/pred_pos are (N, 2) float arrays.
    """
    gt_ids_list: list[np.ndarray] = []
    gt_pos_list: list[np.ndarray] = []
    pred_ids_list: list[np.ndarray] = []
    pred_pos_list: list[np.ndarray] = []

    for dt in range(t_pred):
        fid = pred_start + dt

        frame_gt_ids: list[int] = []
        frame_gt_pos: list[list[float]] = []
        for tid, tdata in gt_tracks.items():
            fid_set = set(tdata["frame_ids"])
            if fid in fid_set:
                idx = tdata["frame_ids"].index(fid)
                frame_gt_ids.append(tid)
                frame_gt_pos.append(tdata["positions"][idx].tolist())

        frame_pred_ids: list[int] = []
        frame_pred_pos: list[list[float]] = []
        for tid, mode_frames in pred_selected.items():
            if fid in mode_frames:
                frame_pred_ids.append(tid)
                frame_pred_pos.append(mode_frames[fid].tolist())

        gt_ids_list.append(np.array(frame_gt_ids, dtype=np.int64))
        gt_pos_list.append(
            np.array(frame_gt_pos, dtype=np.float64).reshape(-1, 2)
            if frame_gt_pos
            else np.zeros((0, 2), dtype=np.float64)
        )
        pred_ids_list.append(np.array(frame_pred_ids, dtype=np.int64))
        pred_pos_list.append(
            np.array(frame_pred_pos, dtype=np.float64).reshape(-1, 2)
            if frame_pred_pos
            else np.zeros((0, 2), dtype=np.float64)
        )

    return gt_ids_list, gt_pos_list, pred_ids_list, pred_pos_list


def eval_sequence(
    gt_ids_per_frame: list[np.ndarray],
    gt_pos_per_frame: list[np.ndarray],
    pred_ids_per_frame: list[np.ndarray],
    pred_pos_per_frame: list[np.ndarray],
    S: float,
    alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    """Core SO-HOTA computation for one evaluation unit.

    This is a direct adaptation of TrackEval's SO_HOTA.eval_sequence(),
    using DotDistance on centroid positions instead of bounding boxes.

    Args:
        gt_ids_per_frame: List of 1D arrays of GT track IDs per frame.
        gt_pos_per_frame: List of (N_gt, 2) arrays of GT positions per frame.
        pred_ids_per_frame: List of 1D arrays of predicted track IDs per frame.
        pred_pos_per_frame: List of (N_pred, 2) arrays of predicted positions.
        S: DotDistance normalization factor.
        alphas: Array of alpha thresholds.

    Returns:
        Dict with SO-HOTA-TP sub-metrics as numpy arrays over alphas.
    """
    n_alphas = len(alphas)
    n_frames = len(gt_ids_per_frame)

    num_gt_dets = sum(len(ids) for ids in gt_ids_per_frame)
    num_pred_dets = sum(len(ids) for ids in pred_ids_per_frame)

    res: dict[str, Any] = {
        "SO-HOTA-TP_TP": np.zeros(n_alphas, dtype=np.float64),
        "SO-HOTA-TP_FN": np.zeros(n_alphas, dtype=np.float64),
        "SO-HOTA-TP_FP": np.zeros(n_alphas, dtype=np.float64),
        "SO-LocA-TP": np.zeros(n_alphas, dtype=np.float64),
        "SO-AssA-TP": np.zeros(n_alphas, dtype=np.float64),
        "SO-AssRe-TP": np.zeros(n_alphas, dtype=np.float64),
        "SO-AssPr-TP": np.zeros(n_alphas, dtype=np.float64),
    }

    if num_pred_dets == 0:
        res["SO-HOTA-TP_FN"] = float(num_gt_dets) * np.ones(n_alphas)
        res["SO-LocA-TP"] = np.ones(n_alphas)
        return _compute_final(res, alphas)
    if num_gt_dets == 0:
        res["SO-HOTA-TP_FP"] = float(num_pred_dets) * np.ones(n_alphas)
        res["SO-LocA-TP"] = np.ones(n_alphas)
        return _compute_final(res, alphas)

    all_gt_ids = sorted(
        set(int(i) for ids in gt_ids_per_frame for i in ids)
    )
    all_pred_ids = sorted(
        set(int(i) for ids in pred_ids_per_frame for i in ids)
    )
    gt_id_map = {gid: idx for idx, gid in enumerate(all_gt_ids)}
    pred_id_map = {pid: idx for idx, pid in enumerate(all_pred_ids)}
    n_gt_ids = len(all_gt_ids)
    n_pred_ids = len(all_pred_ids)

    potential_matches_count = np.zeros((n_gt_ids, n_pred_ids))
    gt_id_count = np.zeros((n_gt_ids, 1))
    pred_id_count = np.zeros((1, n_pred_ids))

    similarity_per_frame: list[np.ndarray] = []
    gt_idx_per_frame: list[np.ndarray] = []
    pred_idx_per_frame: list[np.ndarray] = []

    for t in range(n_frames):
        gt_ids_t = gt_ids_per_frame[t]
        pred_ids_t = pred_ids_per_frame[t]
        gt_pos_t = gt_pos_per_frame[t]
        pred_pos_t = pred_pos_per_frame[t]

        gt_idx_t = np.array([gt_id_map[int(i)] for i in gt_ids_t], dtype=np.int64)
        pred_idx_t = np.array(
            [pred_id_map[int(i)] for i in pred_ids_t], dtype=np.int64
        )
        gt_idx_per_frame.append(gt_idx_t)
        pred_idx_per_frame.append(pred_idx_t)

        similarity = dot_distance_matrix(gt_pos_t, pred_pos_t, S)
        similarity_per_frame.append(similarity)

        if len(gt_idx_t) > 0 and len(pred_idx_t) > 0:
            sim_iou_denom = (
                similarity.sum(0)[np.newaxis, :]
                + similarity.sum(1)[:, np.newaxis]
                - similarity
            )
            sim_iou = np.zeros_like(similarity)
            mask = sim_iou_denom > np.finfo(float).eps
            sim_iou[mask] = similarity[mask] / sim_iou_denom[mask]
            potential_matches_count[
                gt_idx_t[:, np.newaxis], pred_idx_t[np.newaxis, :]
            ] += sim_iou

        gt_id_count[gt_idx_t] += 1
        if len(pred_idx_t) > 0:
            pred_id_count[0, pred_idx_t] += 1

    global_alignment = potential_matches_count / np.maximum(
        1, gt_id_count + pred_id_count - potential_matches_count
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in alphas]

    for t in range(n_frames):
        gt_idx_t = gt_idx_per_frame[t]
        pred_idx_t = pred_idx_per_frame[t]
        similarity = similarity_per_frame[t]

        if len(gt_idx_t) == 0:
            for a in range(n_alphas):
                res["SO-HOTA-TP_FP"][a] += len(pred_idx_t)
            continue
        if len(pred_idx_t) == 0:
            for a in range(n_alphas):
                res["SO-HOTA-TP_FN"][a] += len(gt_idx_t)
            continue

        score_mat = (
            global_alignment[gt_idx_t[:, np.newaxis], pred_idx_t[np.newaxis, :]]
            * similarity
        )
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        for a, alpha in enumerate(alphas):
            matched_mask = (
                similarity[match_rows, match_cols] >= alpha - np.finfo(float).eps
            )
            a_rows = match_rows[matched_mask]
            a_cols = match_cols[matched_mask]
            n_matches = len(a_rows)
            res["SO-HOTA-TP_TP"][a] += n_matches
            res["SO-HOTA-TP_FN"][a] += len(gt_idx_t) - n_matches
            res["SO-HOTA-TP_FP"][a] += len(pred_idx_t) - n_matches
            if n_matches > 0:
                res["SO-LocA-TP"][a] += float(similarity[a_rows, a_cols].sum())
                matches_counts[a][
                    gt_idx_t[a_rows], pred_idx_t[a_cols]
                ] += 1

    for a in range(n_alphas):
        mc = matches_counts[a]
        ass_a = mc / np.maximum(1, gt_id_count + pred_id_count - mc)
        tp_a = res["SO-HOTA-TP_TP"][a]
        res["SO-AssA-TP"][a] = (
            float(np.sum(mc * ass_a)) / max(1, tp_a) if tp_a > 0 else 0.0
        )
        ass_re = mc / np.maximum(1, gt_id_count)
        res["SO-AssRe-TP"][a] = (
            float(np.sum(mc * ass_re)) / max(1, tp_a) if tp_a > 0 else 0.0
        )
        ass_pr = mc / np.maximum(1, pred_id_count)
        res["SO-AssPr-TP"][a] = (
            float(np.sum(mc * ass_pr)) / max(1, tp_a) if tp_a > 0 else 0.0
        )

    res["SO-LocA-TP"] = np.maximum(1e-10, res["SO-LocA-TP"]) / np.maximum(
        1e-10, res["SO-HOTA-TP_TP"]
    )
    return _compute_final(res, alphas)


def _compute_final(res: dict[str, Any], alphas: np.ndarray) -> dict[str, Any]:
    """Compute derived SO-HOTA-TP fields.

    Naming convention follows MVA2025 SO-HOTA:
      SO-HOTA_TP/FN/FP  →  SO-HOTA-TP_TP/FN/FP
      SO-DetA            →  SO-DetA-TP
      SO-AssA            →  SO-AssA-TP
      SO-LocA            →  SO-LocA-TP
      SO-HOTA            →  SO-HOTA-TP
    """
    tp = res["SO-HOTA-TP_TP"]
    fn = res["SO-HOTA-TP_FN"]
    fp = res["SO-HOTA-TP_FP"]

    res["SO-DetRe-TP"] = tp / np.maximum(1, tp + fn)
    res["SO-DetPr-TP"] = tp / np.maximum(1, tp + fp)
    res["SO-DetA-TP"] = tp / np.maximum(1, tp + fn + fp)
    res["SO-HOTA-TP"] = np.sqrt(res["SO-DetA-TP"] * res["SO-AssA-TP"])

    res["SO-HOTA-TP_mean"] = float(np.mean(res["SO-HOTA-TP"]))
    res["SO-DetA-TP_mean"] = float(np.mean(res["SO-DetA-TP"]))
    res["SO-AssA-TP_mean"] = float(np.mean(res["SO-AssA-TP"]))
    res["SO-LocA-TP_mean"] = float(np.mean(res["SO-LocA-TP"]))
    res["SO-DetRe-TP_mean"] = float(np.mean(res["SO-DetRe-TP"]))
    res["SO-DetPr-TP_mean"] = float(np.mean(res["SO-DetPr-TP"]))
    return res


def combine_results(
    all_results: list[dict[str, Any]],
    alphas: np.ndarray,
) -> dict[str, Any]:
    """Combine SO-HOTA-TP results across multiple evaluation windows/sequences.

    Uses detection-count-weighted averaging, matching TrackEval's
    combine_sequences() method.
    """
    res: dict[str, Any] = {
        "SO-HOTA-TP_TP": sum(r["SO-HOTA-TP_TP"] for r in all_results),
        "SO-HOTA-TP_FN": sum(r["SO-HOTA-TP_FN"] for r in all_results),
        "SO-HOTA-TP_FP": sum(r["SO-HOTA-TP_FP"] for r in all_results),
    }

    total_tp = res["SO-HOTA-TP_TP"]
    for field in ("SO-AssA-TP", "SO-AssRe-TP", "SO-AssPr-TP"):
        weighted = sum(r[field] * r["SO-HOTA-TP_TP"] for r in all_results)
        res[field] = weighted / np.maximum(1e-10, total_tp)

    loca_weighted = sum(
        r["SO-LocA-TP"] * r["SO-HOTA-TP_TP"] for r in all_results
    )
    res["SO-LocA-TP"] = np.maximum(1e-10, loca_weighted) / np.maximum(
        1e-10, total_tp
    )

    return _compute_final(res, alphas)


def _restrict_to_frames(
    tracks: dict[int, dict[str, Any]],
    frames: range,
) -> dict[int, dict[str, Any]]:
    """Keep only the observations falling inside `frames`, dropping empty tracks."""
    out: dict[int, dict[str, Any]] = {}
    for tid, tdata in tracks.items():
        keep = [
            (fid, pos)
            for fid, pos in zip(tdata["frame_ids"], tdata["positions"])
            if fid in frames
        ]
        if keep:
            out[tid] = {
                "frame_ids": [f for f, _ in keep],
                "positions": [p for _, p in keep],
            }
    return out


def define_evaluation_windows(
    gt_tracks: dict[int, dict[str, Any]],
    t_obs: int,
    t_pred: int,
) -> list[dict[str, Any]]:
    """Define the evaluation windows of one sequence.

    Windows start at frame 1, the sequence's first image, and advance by
    t_pred while a whole t_obs + t_pred span fits before the last GT frame. A
    window is evaluated if at least one GT track is observed in it.

    The start is fixed to the video, never to the GT: participants do not see
    the GT, so a window grid keyed to its first box would be one they cannot
    reproduce, and every prediction in a sequence whose first bird appears late
    would be keyed to a window that is never scored.

    Returns:
        List of window descriptors with 'obs_start', 'pred_start',
        'pred_end', and 'gt_track_ids' (tracks observed during t_obs).
    """
    all_frames: set[int] = set()
    for tdata in gt_tracks.values():
        all_frames.update(tdata["frame_ids"])
    if not all_frames:
        return []

    max_frame = max(all_frames)

    windows: list[dict[str, Any]] = []
    obs_start = FIRST_FRAME
    while obs_start + t_obs + t_pred - 1 <= max_frame:
        pred_start = obs_start + t_obs
        pred_end = pred_start + t_pred - 1

        observed_tids: list[int] = []
        obs_frames = set(range(obs_start, obs_start + t_obs))
        for tid, tdata in gt_tracks.items():
            track_frames = set(tdata["frame_ids"])
            if track_frames & obs_frames:
                observed_tids.append(tid)

        if observed_tids:
            windows.append(
                {
                    "obs_start": obs_start,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "gt_track_ids": observed_tids,
                }
            )

        obs_start += t_pred

    return windows


def evaluate_so_hota_tp(
    pred_file: str,
    gt_file: str,
    t_obs: int = T_OBS_DEFAULT,
    t_pred: int = T_PRED_DEFAULT,
    S: float | None = None,
    tracks_file: str | None = None,
    match_distance: float | None = None,
) -> dict[str, Any]:
    """Full SO-HOTA-TP evaluation pipeline.

    Args:
        pred_file: Prediction file (frame_id, track_id, cx, cy[, mode_k]).
        gt_file: GT MOT-format file.
        t_obs: Observation window length.
        t_pred: Prediction horizon length.
        S: DotDistance normalization. Computed from GT if None.
        tracks_file: The submission's own tracking output over the observation
            frames, used to associate its arbitrary track ids with GT ids.
            Required whenever submitted ids are not GT ids -- the competition
            case, since participants never receive GT tracks. Omit it only when
            predictions are already keyed by GT id (the GT-input baselines).
        match_distance: Max centre distance for association. Defaults to
            association_max_distance(S), derived from the metric's own weakest
            alpha; pass a value only to override that.

    Returns:
        Dict with SO-HOTA-TP metrics and supplementary ADE/FDE.
    """
    gt_tracks = _load_gt_tracks_with_bbox(gt_file)
    pred_data = _load_predictions(pred_file)

    if S is None:
        S = compute_S_from_gt(gt_file)
    if match_distance is None:
        match_distance = association_max_distance(S)

    tracker_tracks: dict[int, dict[str, Any]] | None = None
    if tracks_file is not None:
        from prediction.data_utils import load_mot_tracks

        tracker_tracks = load_mot_tracks(tracks_file, return_frame_ids=True)

    alphas = np.arange(*ALPHA_RANGE)
    windows = define_evaluation_windows(gt_tracks, t_obs, t_pred)

    if not windows:
        # No window fits in this sequence's GT, so there is nothing to score --
        # distinct from scoring zero. The underscore keys are still returned so
        # evaluate_split can consume the result uniformly.
        return {
            "SO-HOTA-TP": float("nan"),
            "SO-DetA-TP": float("nan"),
            "SO-AssA-TP": float("nan"),
            "SO-LocA-TP": float("nan"),
            "n_windows": 0,
            "n_track_windows": 0,
            "S": S,
            "_window_results": [],
            "_ades": [],
            "_fdes": [],
            "_n_disappeared": 0,
        }

    window_results: list[dict[str, Any]] = []
    all_ades: list[float] = []
    all_fdes: list[float] = []
    n_disappeared_tracks = 0
    n_total_track_windows = 0

    for win in windows:
        pred_start = win["pred_start"]
        pred_end = win["pred_end"]
        pred_frames = list(range(pred_start, pred_end + 1))

        win_gt_tracks: dict[int, dict[str, Any]] = {}
        for tid in win["gt_track_ids"]:
            if tid in gt_tracks:
                win_gt_tracks[tid] = gt_tracks[tid]

        # Predictions for this window only. A submission keyed by window_start
        # is selected here; a legacy single-window file lands under -1 and is
        # used for every window, which is why it only makes sense on fixtures.
        win_pred_data = pred_data.get(win["obs_start"], pred_data.get(-1, {}))

        # Associate per window, not once per sequence: a tracker may switch ids
        # mid-sequence, so one global mapping would be wrong for some windows.
        # Restrict both sides to this window's observation frames -- matching on
        # the predicted future would let a submission be scored against whichever
        # GT track its guess happened to land on.
        if tracker_tracks is not None:
            from evaluation.eval_prediction import associate_tracks

            obs_range = range(win["obs_start"], pred_start)
            mapping = associate_tracks(
                _restrict_to_frames(tracker_tracks, obs_range),
                _restrict_to_frames(win_gt_tracks, obs_range),
                match_distance,
            )
            win_pred_data = {
                mapping[tid]: modes
                for tid, modes in win_pred_data.items()
                if tid in mapping
            }

        pred_selected: dict[int, dict[int, np.ndarray]] = {}
        for tid in win["gt_track_ids"]:
            n_total_track_windows += 1
            gt_frame_pos: dict[int, np.ndarray] = {}
            if tid in gt_tracks:
                tdata = gt_tracks[tid]
                for fid, pos in zip(tdata["frame_ids"], tdata["positions"]):
                    if pred_start <= fid <= pred_end:
                        gt_frame_pos[fid] = pos

            has_disappearance = len(gt_frame_pos) < t_pred
            if has_disappearance:
                n_disappeared_tracks += 1

            if tid in win_pred_data:
                selected = select_best_mode(win_pred_data[tid], gt_frame_pos, S)
                pred_selected[tid] = selected

                visible_preds = []
                visible_gts = []
                for fid in sorted(gt_frame_pos.keys()):
                    if fid in selected:
                        visible_preds.append(selected[fid])
                        visible_gts.append(gt_frame_pos[fid])

                if visible_preds:
                    preds_arr = np.array(visible_preds)
                    gts_arr = np.array(visible_gts)
                    dists = np.linalg.norm(preds_arr - gts_arr, axis=1)
                    all_ades.append(float(np.mean(dists)))
                    all_fdes.append(float(dists[-1]))

        gt_ids_pf, gt_pos_pf, pred_ids_pf, pred_pos_pf = (
            _build_window_frame_data(
                win_gt_tracks, pred_selected, pred_start, t_pred
            )
        )

        win_result = eval_sequence(
            gt_ids_pf,
            gt_pos_pf,
            pred_ids_pf,
            pred_pos_pf,
            S,
            alphas,
        )
        window_results.append(win_result)

    combined = combine_results(window_results, alphas)

    disappearance_rate = (
        n_disappeared_tracks / n_total_track_windows
        if n_total_track_windows > 0
        else 0.0
    )

    return {
        "SO-HOTA-TP": combined["SO-HOTA-TP_mean"],
        "SO-DetA-TP": combined["SO-DetA-TP_mean"],
        "SO-AssA-TP": combined["SO-AssA-TP_mean"],
        "SO-LocA-TP": combined["SO-LocA-TP_mean"],
        "SO-DetRe-TP": combined["SO-DetRe-TP_mean"],
        "SO-DetPr-TP": combined["SO-DetPr-TP_mean"],
        "SO-HOTA-TP_per_alpha": combined["SO-HOTA-TP"].tolist(),
        "SO-DetA-TP_per_alpha": combined["SO-DetA-TP"].tolist(),
        "SO-AssA-TP_per_alpha": combined["SO-AssA-TP"].tolist(),
        "alphas": alphas.tolist(),
        "minADE": float(np.mean(all_ades)) if all_ades else float("nan"),
        "minFDE": float(np.mean(all_fdes)) if all_fdes else float("nan"),
        "S": S,
        "n_windows": len(windows),
        "n_track_windows": n_total_track_windows,
        "disappearance_rate": disappearance_rate,
        "_window_results": window_results,
        "_ades": all_ades,
        "_fdes": all_fdes,
        "_n_disappeared": n_disappeared_tracks,
    }


def evaluate_split(
    pred_dir: Path | str,
    gt_dir: Path | str,
    tracks_dir: Path | str | None = None,
    t_obs: int = T_OBS_DEFAULT,
    t_pred: int = T_PRED_DEFAULT,
    S: float | None = None,
    match_distance: float | None = None,
) -> dict[str, Any]:
    """Score a whole split: one prediction file per sequence, combined.

    This is what a leaderboard needs. `evaluate_so_hota_tp` scores one sequence;
    scoring a split is not the mean of those, because sequences contribute
    unequally -- SO-HOTA weights association by true-positive count, exactly as
    TrackEval's combine_sequences does. Averaging per-sequence scores would let
    a sequence holding two windows count as much as one holding forty.

    S is computed once over the whole split and shared, so the DotD scale does
    not shift between sequences.

    Args:
        pred_dir: directory of `{seq}.txt` predictions.
        gt_dir: directory of `{seq}.txt` MOT ground truth. Sequences here define
            what is scored; a missing prediction file scores zero rather than
            being skipped.
        tracks_dir: directory of `{seq}.txt` tracking output, for id association.
        t_obs, t_pred, S, match_distance: as `evaluate_so_hota_tp`.

    Returns:
        Combined metrics, plus `per_sequence` and the sequence counts.
    """
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    tracks_dir = Path(tracks_dir) if tracks_dir else None

    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise ValueError(f"no ground-truth sequences in {gt_dir}")

    if S is None:
        areas: list[float] = []
        for g in gt_files:
            for line in g.read_text().splitlines():
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    areas.append(float(parts[4]) * float(parts[5]))
        if not areas:
            raise ValueError(f"no boxes found in {gt_dir}")
        S = float(np.sqrt(np.mean(areas)))
    if match_distance is None:
        match_distance = association_max_distance(S)

    alphas = np.arange(*ALPHA_RANGE)
    window_results: list[dict[str, Any]] = []
    ades: list[float] = []
    fdes: list[float] = []
    per_sequence: dict[str, Any] = {}
    n_missing = n_disappeared = n_track_windows = 0

    for g in gt_files:
        seq = g.stem
        p = pred_dir / f"{seq}.txt"
        missing = not p.exists()
        if missing:
            # Score it as a total miss rather than skipping: its windows exist
            # in the GT, so dropping them would let a submission raise its score
            # by omitting the sequences it finds hard. An empty prediction file
            # produces the same windows with every GT detection unmatched, which
            # is exactly a run of false negatives.
            n_missing += 1
        t = tracks_dir / f"{seq}.txt" if tracks_dir else None
        with tempfile.TemporaryDirectory() as tmp:
            if missing:
                # Not written next to the submission: that directory may be
                # read-only on a scoring server.
                p = Path(tmp) / f"{seq}.txt"
                p.write_text("")
            r = evaluate_so_hota_tp(
                str(p), str(g),
                t_obs=t_obs, t_pred=t_pred, S=S,
                tracks_file=str(t) if t and t.exists() else None,
                match_distance=match_distance,
            )
        window_results.extend(r.pop("_window_results"))
        ades.extend(r.pop("_ades"))
        fdes.extend(r.pop("_fdes"))
        n_disappeared += r.pop("_n_disappeared")
        n_track_windows += r["n_track_windows"]
        per_sequence[seq] = {
            k: v for k, v in r.items() if not k.endswith("_per_alpha")
        }
        if missing:
            # Only a sequence that actually had windows was really missed; one
            # with none is unscorable in the GT and costs the submission nothing.
            per_sequence[seq]["missing"] = True
            if r["n_windows"] == 0:
                n_missing -= 1
                per_sequence[seq]["no_windows"] = True

    if not window_results:
        return {
            "SO-HOTA-TP": 0.0, "SO-DetA-TP": 0.0, "SO-AssA-TP": 0.0,
            "SO-LocA-TP": 0.0, "SO-DetRe-TP": 0.0, "SO-DetPr-TP": 0.0,
            "minADE": float("nan"), "minFDE": float("nan"),
            "S": S, "n_windows": 0, "n_sequences": len(gt_files),
            "n_missing_sequences": n_missing, "disappearance_rate": 0.0,
            "per_sequence": per_sequence,
        }

    c = combine_results(window_results, alphas)
    return {
        "SO-HOTA-TP": c["SO-HOTA-TP_mean"],
        "SO-DetA-TP": c["SO-DetA-TP_mean"],
        "SO-AssA-TP": c["SO-AssA-TP_mean"],
        "SO-LocA-TP": c["SO-LocA-TP_mean"],
        "SO-DetRe-TP": c["SO-DetRe-TP_mean"],
        "SO-DetPr-TP": c["SO-DetPr-TP_mean"],
        "minADE": float(np.mean(ades)) if ades else float("nan"),
        "minFDE": float(np.mean(fdes)) if fdes else float("nan"),
        "S": S,
        "n_windows": len(window_results),
        "n_sequences": len(gt_files),
        "n_missing_sequences": n_missing,
        "disappearance_rate": (
            n_disappeared / n_track_windows if n_track_windows else 0.0
        ),
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory predictions with SO-HOTA-TP"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Prediction file"
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Ground truth MOT file"
    )
    parser.add_argument(
        "--t-obs",
        type=int,
        default=T_OBS_DEFAULT,
        help=f"Observation frames (default: {T_OBS_DEFAULT})",
    )
    parser.add_argument(
        "--t-pred",
        type=int,
        default=T_PRED_DEFAULT,
        help=f"Prediction frames (default: {T_PRED_DEFAULT})",
    )
    parser.add_argument(
        "--S",
        type=float,
        default=None,
        help="DotDistance normalization factor (computed from GT if omitted)",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default=None,
        help="The submission's tracking output over the observation frames, for "
             "associating its track ids with GT ids. Required unless the "
             "prediction file is already keyed by GT id.",
    )
    parser.add_argument(
        "--match-distance",
        type=float,
        default=None,
        help="Max centre distance for track association (px). Default: derived "
             "from S and the metric's weakest alpha (~3x the mean box).",
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Save results to JSON"
    )
    args = parser.parse_args()

    results = evaluate_so_hota_tp(
        args.pred,
        args.gt,
        t_obs=args.t_obs,
        t_pred=args.t_pred,
        S=args.S,
        tracks_file=args.tracks,
        match_distance=args.match_distance,
    )

    print(f"\nSO-HOTA-TP Evaluation (n_windows={results['n_windows']}, S={results['S']:.2f})")
    print(f"  SO-HOTA-TP:  {results['SO-HOTA-TP']:.4f}")
    print(f"  SO-DetA-TP:  {results['SO-DetA-TP']:.4f}")
    print(f"  SO-AssA-TP:  {results['SO-AssA-TP']:.4f}")
    print(f"  SO-LocA-TP:  {results['SO-LocA-TP']:.4f}")
    print(f"  SO-DetRe-TP: {results['SO-DetRe-TP']:.4f}")
    print(f"  SO-DetPr-TP: {results['SO-DetPr-TP']:.4f}")
    print(f"  ---")
    print(f"  minADE (ref): {results['minADE']:.4f} px")
    print(f"  minFDE (ref): {results['minFDE']:.4f} px")
    print(f"  Disappearance rate: {results['disappearance_rate']:.1%}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            # Underscore keys carry raw per-window state for evaluate_split to
            # combine; they hold numpy arrays and are not JSON-serialisable.
            json.dump(
                {k: v for k, v in results.items() if not k.startswith("_")},
                f, indent=2,
            )


if __name__ == "__main__":
    main()
