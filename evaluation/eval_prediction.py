"""Evaluation metrics for trajectory prediction.

Implements minADE_K, minFDE_K and Miss Rate as defined in the SMOTP4SB Track 2
specification, scored from MOT-format files.

Prediction files are `frame_id, track_id, cx, cy[, mode_k]`. The fifth column is
optional; without it each track carries one hypothesis and the reported figure is
ADE_1, which the CLI states explicitly rather than labelling it minADE_K.

Windows the predictor did not cover are counted and reported as a coverage
figure. Scoring only the windows a model managed to reach makes a model with poor
recall look accurate, which is the failure this cascade is most prone to.

For scoring arrays already in memory -- which is how the baselines in
`prediction/` report -- use `metrics()` in `prediction.eval_constant_velocity`.

Usage:
    python -m evaluation.eval_prediction --pred pred.txt --gt gt.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error between one prediction and ground truth.

    Args:
        pred: (T_pred, 2) predicted positions.
        gt: (T_pred, 2) ground truth positions.

    Returns:
        Mean L2 distance over all timesteps.
    """
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error.

    Args:
        pred: (T_pred, 2) predicted positions.
        gt: (T_pred, 2) ground truth positions.

    Returns:
        L2 distance at the final timestep.
    """
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def compute_min_ade_fde(
    preds: np.ndarray,
    gt: np.ndarray,
) -> tuple[float, float]:
    """Compute minADE_K and minFDE_K.

    Args:
        preds: (K, T_pred, 2) K predicted trajectories.
        gt: (T_pred, 2) ground truth trajectory.

    Returns:
        (minADE, minFDE) over K samples.
    """
    ades = [compute_ade(preds[k], gt) for k in range(len(preds))]
    fdes = [compute_fde(preds[k], gt) for k in range(len(preds))]
    best_k_ade = int(np.argmin(ades))
    return ades[best_k_ade], fdes[best_k_ade]


def compute_miss_rate(fde_values: np.ndarray, threshold: float) -> float:
    """Fraction of samples where FDE exceeds threshold.

    Args:
        fde_values: (N,) array of FDE values.
        threshold: Miss threshold in pixels.

    Returns:
        Miss rate in [0, 1].
    """
    return float(np.mean(fde_values > threshold))


def associate_tracks(
    tracker_tracks: dict,
    gt_tracks: dict,
    max_distance: float,
    min_overlap: int = 5,
) -> dict[int, int]:
    """Map tracker track ids onto GT track ids by Hungarian matching.

    A tracker assigns arbitrary ids, so they carry no relation to GT ids. Cost is
    the mean centre distance over frames the two tracks share; pairs sharing fewer
    than min_overlap frames, or averaging more than max_distance apart, are not
    matched. Matching uses observed positions only -- never the predicted futures,
    which are what the metrics score.

    Returns:
        Dict mapping tracker_id -> gt_id for matched pairs only.
    """
    from scipy.optimize import linear_sum_assignment

    t_ids = sorted(tracker_tracks)
    g_ids = sorted(gt_tracks)
    if not t_ids or not g_ids:
        return {}

    UNMATCHED = 1e9
    cost = np.full((len(t_ids), len(g_ids)), UNMATCHED)
    for i, tid in enumerate(t_ids):
        t_by_frame = dict(zip(tracker_tracks[tid]["frame_ids"], tracker_tracks[tid]["positions"]))
        for j, gid in enumerate(g_ids):
            g_by_frame = dict(zip(gt_tracks[gid]["frame_ids"], gt_tracks[gid]["positions"]))
            shared = t_by_frame.keys() & g_by_frame.keys()
            if len(shared) < min_overlap:
                continue
            d = np.mean([np.linalg.norm(t_by_frame[f] - g_by_frame[f]) for f in shared])
            if d <= max_distance:
                cost[i, j] = d

    rows, cols = linear_sum_assignment(cost)
    return {
        t_ids[i]: g_ids[j]
        for i, j in zip(rows, cols)
        if cost[i, j] < UNMATCHED
    }


def evaluate_prediction_file(
    pred_file: str,
    gt_file: str,
    t_obs: int,
    t_pred: int,
    miss_threshold: float,
    tracks_file: str | None = None,
    match_distance: float = 100.0,
) -> dict[str, float]:
    """Evaluate a prediction file against ground truth.

    Both files are in MOT format. GT is the full tracking ground truth;
    predictions contain only future positions.

    Args:
        pred_file: Path to predicted trajectories (frame_id, track_id, cx, cy).
        gt_file: Path to GT MOT file.
        t_obs: Observation frames (for window alignment).
        t_pred: Prediction frames.
        miss_threshold: Miss rate threshold in pixels.
        tracks_file: Tracker output the predictions came from. Required whenever the
            predictions derive from a tracker rather than from GT, so its arbitrary
            ids can be associated with GT ids. Without it, ids are compared directly
            and almost nothing matches.
        match_distance: Maximum mean centre distance (px) for a valid association.

    Returns:
        Dict with ADE, FDE, MissRate.
    """
    from prediction.data_utils import load_mot_tracks, extract_windows

    gt_tracks = load_mot_tracks(gt_file, return_frame_ids=True)

    # Format: frame_id, track_id, cx, cy[, mode_k]. The optional fifth column is
    # the mode index, so one file can carry K hypotheses per track and minADE_K
    # means what it says. Files written without it are read as a single mode, in
    # which case minADE_K degenerates to ADE_1 -- which is the honest reading of
    # a unimodal prediction, not a K-mode score.
    pred_data: dict[int, dict[int, dict[int, np.ndarray]]] = {}
    with open(pred_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            fid, tid = int(parts[0]), int(float(parts[1]))
            cx, cy = float(parts[2]), float(parts[3])
            k = int(float(parts[4])) if len(parts) > 4 else 0
            pred_data.setdefault(tid, {}).setdefault(k, {})[fid] = np.array([cx, cy])

    n_matched = None
    if tracks_file is not None:
        tracker_tracks = load_mot_tracks(tracks_file, return_frame_ids=True)
        mapping = associate_tracks(tracker_tracks, gt_tracks, match_distance)
        n_matched = len(mapping)
        # Re-key predictions onto GT ids; unmatched tracker ids are dropped rather
        # than scored against an arbitrary GT track.
        pred_data = {
            mapping[tid]: preds
            for tid, preds in pred_data.items()
            if tid in mapping
        }

    all_ades: list[float] = []
    all_fdes: list[float] = []
    mode_counts: set[int] = set()
    n_untracked = 0      # GT track the predictor never produced anything for
    n_short = 0          # prediction stopped before covering the full horizon

    for tid, track_data in gt_tracks.items():
        positions = track_data["positions"]
        frame_ids = track_data["frame_ids"]
        windows = extract_windows(positions, frame_ids, t_obs, t_pred, stride=t_pred)

        for obs, gt_future, meta in windows:
            pred_start = meta["obs_end_frame"] + 1
            if tid not in pred_data:
                n_untracked += 1
                continue

            modes = []
            for k in sorted(pred_data[tid]):
                by_frame = pred_data[tid][k]
                seq = [by_frame[pred_start + t] for t in range(t_pred)
                       if pred_start + t in by_frame]
                if len(seq) == t_pred:
                    modes.append(np.asarray(seq))

            if not modes:
                # Counted, never silently skipped. A window the predictor could
                # not cover for its full horizon is a real failure, and dropping
                # it without a tally flatters every metric reported below.
                n_short += 1
                continue

            mode_counts.add(len(modes))
            ade, fde = compute_min_ade_fde(np.stack(modes), gt_future)
            all_ades.append(ade)
            all_fdes.append(fde)

    # Coverage is part of the result, not a footnote: a predictor that scores
    # well on the few windows it managed to cover is not a better predictor.
    n_windows = len(all_ades) + n_untracked + n_short
    coverage = len(all_ades) / n_windows if n_windows else float("nan")

    if not all_ades:
        return {
            "ADE": float("nan"), "FDE": float("nan"), "MissRate": float("nan"),
            "n_samples": 0, "n_matched_tracks": n_matched, "K": 0,
            "n_gt_windows": n_windows, "n_dropped_untracked": n_untracked,
            "n_dropped_short": n_short, "coverage": coverage,
        }

    fde_arr = np.array(all_fdes)
    return {
        "ADE": float(np.mean(all_ades)),
        "FDE": float(np.mean(all_fdes)),
        "MissRate": compute_miss_rate(fde_arr, miss_threshold),
        "n_samples": len(all_ades),
        "n_matched_tracks": n_matched,
        "K": max(mode_counts) if mode_counts else 0,
        "n_gt_windows": n_windows,
        "n_dropped_untracked": n_untracked,
        "n_dropped_short": n_short,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trajectory predictions")
    parser.add_argument("--pred", type=str, required=True, help="Prediction file")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth MOT file")
    parser.add_argument("--t-obs", type=int, default=60, help="Observation frames (task: 60)")
    parser.add_argument("--t-pred", type=int, default=150, help="Prediction frames (task: 150)")
    parser.add_argument("--miss-threshold", type=float, default=48.0,
                        help="Miss rate threshold in pixels (task: 48)")
    parser.add_argument("--tracks", type=str, default=None,
                        help="Tracker output the predictions came from. Required when predictions "
                             "derive from a tracker, so its ids can be matched to GT ids.")
    parser.add_argument("--match-distance", type=float, default=100.0,
                        help="Max mean centre distance (px) for a track association")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = evaluate_prediction_file(
        args.pred, args.gt, args.t_obs, args.t_pred, args.miss_threshold,
        tracks_file=args.tracks, match_distance=args.match_distance,
    )

    K = results["K"]
    sub = K if K else "K"          # nothing scored: no K to report
    print(f"Prediction Evaluation Results (n={results['n_samples']}, K={K})")
    if results.get("n_matched_tracks") is not None:
        print(f"  Matched tracks:  {results['n_matched_tracks']}")
    print(f"  minADE_{sub}:       {results['ADE']:.4f} px")
    print(f"  minFDE_{sub}:       {results['FDE']:.4f} px")
    print(f"  MissRate:        {results['MissRate']:.4f} (threshold={args.miss_threshold}px)")
    print(f"  Coverage:        {results['coverage']:.1%} of {results['n_gt_windows']} GT windows")
    print(f"    dropped, no prediction for the track: {results['n_dropped_untracked']}")
    print(f"    dropped, horizon not covered:         {results['n_dropped_short']}")
    if K == 1:
        print("  NOTE: the prediction file carries one mode per track, so this is "
              "ADE_1/FDE_1.\n        Add a fifth column (mode index) to score minADE_K.")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
