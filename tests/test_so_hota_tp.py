"""Tests for SO-HOTA-TP evaluation metric.

Covers:
  1. DotDistance computation
  2. Best-of-K mode selection
  3. Perfect prediction → SO-HOTA-TP = 1.0
  4. Disappearance handling (FP/FN)
  5. Identity swap detection (AssA < 1)
  6. Edge cases (empty predictions, empty GT)
  7. Consistency with MVA2025 SO-HOTA on tracking-equivalent input
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.so_hota_tp import (
    _load_predictions,
    association_max_distance,
    combine_results,
    compute_S_from_gt,
    define_evaluation_windows,
    dot_distance_matrix,
    eval_sequence,
    select_best_mode,
)

ALPHAS = np.arange(0.05, 0.99, 0.05)
S_TEST = 20.0


# --------------------------------------------------------------------------- #
# 1. DotDistance
# --------------------------------------------------------------------------- #
class TestDotDistance:
    def test_identical_positions_give_score_one(self):
        pos = np.array([[100.0, 200.0], [300.0, 400.0]])
        sim = dot_distance_matrix(pos, pos, S_TEST)
        np.testing.assert_allclose(np.diag(sim), 1.0)

    def test_far_positions_give_score_near_zero(self):
        gt = np.array([[100.0, 100.0]])
        pred = np.array([[10000.0, 10000.0]])
        sim = dot_distance_matrix(gt, pred, S_TEST)
        assert sim[0, 0] < 1e-10

    def test_known_distance(self):
        gt = np.array([[0.0, 0.0]])
        pred = np.array([[S_TEST, 0.0]])
        sim = dot_distance_matrix(gt, pred, S_TEST)
        np.testing.assert_allclose(sim[0, 0], np.exp(-1.0), rtol=1e-10)

    def test_empty_inputs(self):
        empty = np.zeros((0, 2))
        pos = np.array([[1.0, 2.0]])
        assert dot_distance_matrix(empty, pos, S_TEST).shape == (0, 1)
        assert dot_distance_matrix(pos, empty, S_TEST).shape == (1, 0)
        assert dot_distance_matrix(empty, empty, S_TEST).shape == (0, 0)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        gt = rng.uniform(0, 1000, (5, 2))
        pred = rng.uniform(0, 1000, (3, 2))
        sim_ab = dot_distance_matrix(gt, pred, S_TEST)
        sim_ba = dot_distance_matrix(pred, gt, S_TEST)
        np.testing.assert_allclose(sim_ab, sim_ba.T, rtol=1e-10)


# --------------------------------------------------------------------------- #
# 2. Best-of-K mode selection
# --------------------------------------------------------------------------- #
class TestBestOfK:
    def test_selects_closest_mode(self):
        gt_frames = {
            61: np.array([100.0, 100.0]),
            62: np.array([101.0, 101.0]),
        }
        modes = {
            0: {61: np.array([200.0, 200.0]), 62: np.array([201.0, 201.0])},
            1: {61: np.array([100.5, 100.5]), 62: np.array([101.5, 101.5])},
        }
        selected = select_best_mode(modes, gt_frames, S_TEST)
        assert 61 in selected
        np.testing.assert_allclose(selected[61], [100.5, 100.5])

    def test_single_mode_always_selected(self):
        gt_frames = {61: np.array([100.0, 100.0])}
        modes = {0: {61: np.array([500.0, 500.0])}}
        selected = select_best_mode(modes, gt_frames, S_TEST)
        np.testing.assert_allclose(selected[61], [500.0, 500.0])

    def test_empty_modes_returns_empty(self):
        gt_frames = {61: np.array([100.0, 100.0])}
        selected = select_best_mode({}, gt_frames, S_TEST)
        assert selected == {}


# --------------------------------------------------------------------------- #
# 3. Perfect prediction → SO-HOTA-TP = 1.0
# --------------------------------------------------------------------------- #
class TestPerfectPrediction:
    def test_perfect_single_track(self):
        """Identical GT and prediction → all metrics = 1.0."""
        t_pred = 10
        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[100.0 + t, 200.0 + t]]) for t in range(t_pred)]
        pred_ids = [np.array([1]) for _ in range(t_pred)]
        pred_pos = [np.array([[100.0 + t, 200.0 + t]]) for t in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["SO-DetA-TP"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["SO-AssA-TP"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["SO-LocA-TP"], 1.0, atol=1e-10)

    def test_perfect_multi_track(self):
        """Multiple tracks with perfect predictions."""
        t_pred = 10
        gt_ids = [np.array([1, 2]) for _ in range(t_pred)]
        gt_pos = [
            np.array([[100.0 + t, 200.0], [500.0 + t, 600.0]])
            for t in range(t_pred)
        ]
        pred_ids = [np.array([1, 2]) for _ in range(t_pred)]
        pred_pos = [
            np.array([[100.0 + t, 200.0], [500.0 + t, 600.0]])
            for t in range(t_pred)
        ]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["SO-AssA-TP"], 1.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# 4. Disappearance handling
# --------------------------------------------------------------------------- #
class TestDisappearance:
    def test_gt_disappears_prediction_continues_is_fp(self):
        """Bird disappears at frame 5, but prediction continues → FP."""
        t_pred = 10
        gt_ids = [np.array([1]) for _ in range(5)] + [
            np.array([], dtype=np.int64) for _ in range(5)
        ]
        gt_pos = [np.array([[100.0, 200.0]]) for _ in range(5)] + [
            np.zeros((0, 2)) for _ in range(5)
        ]
        pred_ids = [np.array([1]) for _ in range(t_pred)]
        pred_pos = [np.array([[100.0, 200.0]]) for _ in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        assert result["SO-HOTA-TP_mean"] < 1.0
        assert np.all(result["SO-HOTA-TP_FP"] > 0)
        assert np.all(result["SO-HOTA-TP_TP"] > 0)

    def test_gt_disappears_prediction_stops_is_correct(self):
        """Bird disappears and predictor correctly stops → no FP for absent frames."""
        t_pred = 10
        gt_ids = [np.array([1]) for _ in range(5)] + [
            np.array([], dtype=np.int64) for _ in range(5)
        ]
        gt_pos = [np.array([[100.0, 200.0]]) for _ in range(5)] + [
            np.zeros((0, 2)) for _ in range(5)
        ]
        pred_ids = [np.array([1]) for _ in range(5)] + [
            np.array([], dtype=np.int64) for _ in range(5)
        ]
        pred_pos = [np.array([[100.0, 200.0]]) for _ in range(5)] + [
            np.zeros((0, 2)) for _ in range(5)
        ]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP"], 1.0, atol=1e-10)
        np.testing.assert_allclose(result["SO-HOTA-TP_FP"], 0.0, atol=1e-10)

    def test_no_prediction_for_visible_bird_is_fn(self):
        """Bird is visible but predictor provides nothing → FN."""
        t_pred = 10
        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[100.0, 200.0]]) for _ in range(t_pred)]
        pred_ids = [np.array([], dtype=np.int64) for _ in range(t_pred)]
        pred_pos = [np.zeros((0, 2)) for _ in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        assert result["SO-HOTA-TP_mean"] == 0.0
        np.testing.assert_allclose(result["SO-HOTA-TP_FN"], t_pred)
        np.testing.assert_allclose(result["SO-HOTA-TP_TP"], 0.0)


# --------------------------------------------------------------------------- #
# 5. Identity swap detection
# --------------------------------------------------------------------------- #
class TestAssociation:
    def test_id_swap_reduces_assa(self):
        """Two tracks that swap predictions mid-way → AssA < 1.0."""
        t_pred = 20
        gt_ids_list = [np.array([1, 2]) for _ in range(t_pred)]
        gt_pos_list = [
            np.array([[100.0, 100.0], [500.0, 500.0]]) for _ in range(t_pred)
        ]

        pred_ids_list = []
        pred_pos_list = []
        for t in range(t_pred):
            if t < 10:
                pred_ids_list.append(np.array([1, 2]))
                pred_pos_list.append(
                    np.array([[100.0, 100.0], [500.0, 500.0]])
                )
            else:
                pred_ids_list.append(np.array([1, 2]))
                pred_pos_list.append(
                    np.array([[500.0, 500.0], [100.0, 100.0]])
                )

        result = eval_sequence(
            gt_ids_list, gt_pos_list,
            pred_ids_list, pred_pos_list,
            S_TEST, ALPHAS,
        )

        assert result["SO-DetA-TP_mean"] > 0.9
        assert result["SO-AssA-TP_mean"] < 0.9
        assert result["SO-HOTA-TP_mean"] < result["SO-DetA-TP_mean"]

    def test_consistent_ids_give_high_assa(self):
        """Correct but imperfect predictions with consistent IDs → high AssA."""
        t_pred = 20
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2.0, (t_pred, 2))

        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[300.0, 300.0]]) for _ in range(t_pred)]
        pred_ids = [np.array([1]) for _ in range(t_pred)]
        pred_pos = [np.array([[300.0 + noise[t, 0], 300.0 + noise[t, 1]]]) for t in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        assert result["SO-AssA-TP_mean"] > 0.85


# --------------------------------------------------------------------------- #
# 6. Edge cases
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_empty_gt_all_fp(self):
        t_pred = 5
        gt_ids = [np.array([], dtype=np.int64) for _ in range(t_pred)]
        gt_pos = [np.zeros((0, 2)) for _ in range(t_pred)]
        pred_ids = [np.array([1]) for _ in range(t_pred)]
        pred_pos = [np.array([[100.0, 100.0]]) for _ in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP_FP"], t_pred)
        np.testing.assert_allclose(result["SO-HOTA-TP_TP"], 0.0)

    def test_empty_pred_all_fn(self):
        t_pred = 5
        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[100.0, 100.0]]) for _ in range(t_pred)]
        pred_ids = [np.array([], dtype=np.int64) for _ in range(t_pred)]
        pred_pos = [np.zeros((0, 2)) for _ in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP_FN"], t_pred)
        np.testing.assert_allclose(result["SO-HOTA-TP_TP"], 0.0)

    def test_both_empty_returns_zeros(self):
        t_pred = 5
        gt_ids = [np.array([], dtype=np.int64) for _ in range(t_pred)]
        gt_pos = [np.zeros((0, 2)) for _ in range(t_pred)]
        pred_ids = [np.array([], dtype=np.int64) for _ in range(t_pred)]
        pred_pos = [np.zeros((0, 2)) for _ in range(t_pred)]

        result = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)

        np.testing.assert_allclose(result["SO-HOTA-TP_TP"], 0.0)
        np.testing.assert_allclose(result["SO-HOTA-TP_FN"], 0.0)
        np.testing.assert_allclose(result["SO-HOTA-TP_FP"], 0.0)


# --------------------------------------------------------------------------- #
# 7. Combine results
# --------------------------------------------------------------------------- #
class TestCombine:
    def test_combine_perfect_windows(self):
        """Two perfect windows combined → still perfect."""
        t_pred = 5
        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[100.0, 200.0]]) for _ in range(t_pred)]
        pred_ids = gt_ids
        pred_pos = gt_pos

        r1 = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)
        r2 = eval_sequence(gt_ids, gt_pos, pred_ids, pred_pos, S_TEST, ALPHAS)
        combined = combine_results([r1, r2], ALPHAS)

        np.testing.assert_allclose(combined["SO-HOTA-TP"], 1.0, atol=1e-10)

    def test_combine_mixed_quality(self):
        """Perfect + empty-pred combined → intermediate score."""
        t_pred = 5
        gt_ids = [np.array([1]) for _ in range(t_pred)]
        gt_pos = [np.array([[100.0, 200.0]]) for _ in range(t_pred)]

        r_perfect = eval_sequence(gt_ids, gt_pos, gt_ids, gt_pos, S_TEST, ALPHAS)

        r_empty = eval_sequence(
            gt_ids,
            gt_pos,
            [np.array([], dtype=np.int64)] * t_pred,
            [np.zeros((0, 2))] * t_pred,
            S_TEST,
            ALPHAS,
        )

        combined = combine_results([r_perfect, r_empty], ALPHAS)
        assert 0.0 < combined["SO-HOTA-TP_mean"] < 1.0


# --------------------------------------------------------------------------- #
# 8. S computation
# --------------------------------------------------------------------------- #
class TestComputeS:
    def test_compute_s_from_file(self, tmp_path):
        gt_file = tmp_path / "gt.txt"
        lines = [
            "1,1,100,100,20,30,1,1,1",
            "1,2,200,200,40,40,1,1,1",
            "2,1,101,101,20,30,1,1,1",
        ]
        gt_file.write_text("\n".join(lines))
        S = compute_S_from_gt(gt_file)
        areas = [20 * 30, 40 * 40, 20 * 30]
        expected = np.sqrt(np.mean(areas))
        np.testing.assert_allclose(S, expected, rtol=1e-10)

    def test_empty_file_raises(self, tmp_path):
        gt_file = tmp_path / "empty.txt"
        gt_file.write_text("")
        with pytest.raises(ValueError, match="No valid"):
            compute_S_from_gt(gt_file)


# --------------------------------------------------------------------------- #
# 9. Window definition
# --------------------------------------------------------------------------- #
class TestWindowDefinition:
    def test_basic_windows(self):
        tracks = {
            1: {
                "frame_ids": list(range(1, 301)),
                "positions": np.zeros((300, 2)),
                "bboxes": np.zeros((300, 4)),
            }
        }
        windows = define_evaluation_windows(tracks, t_obs=60, t_pred=150)
        assert len(windows) == 1
        assert windows[0]["obs_start"] == 1
        assert windows[0]["pred_start"] == 61
        assert windows[0]["pred_end"] == 210

    def test_multiple_windows(self):
        tracks = {
            1: {
                "frame_ids": list(range(1, 601)),
                "positions": np.zeros((600, 2)),
                "bboxes": np.zeros((600, 4)),
            }
        }
        windows = define_evaluation_windows(tracks, t_obs=60, t_pred=150)
        assert len(windows) >= 2
        assert windows[1]["obs_start"] == 151

    def test_windows_start_at_frame_one_when_gt_starts_late(self):
        """Windows are cut from the video, not from the hidden GT.

        Participants never see GT, so they cannot know when its first box is.
        In 28 train and 7 pub_test sequences the first bird appears after
        frame 1; keying windows to the GT would shift every window there and
        leave a correct submission unscored.
        """
        tracks = {
            1: {
                "frame_ids": list(range(79, 601)),
                "positions": np.zeros((522, 2)),
                "bboxes": np.zeros((522, 4)),
            }
        }
        starts = [w["obs_start"] for w in
                  define_evaluation_windows(tracks, t_obs=60, t_pred=150)]
        assert starts[0] == 151 and all((s - 1) % 150 == 0 for s in starts)

    def test_window_with_no_observed_gt_is_skipped(self):
        """Frame 1-60 hold no GT here, so the first window has nothing to score."""
        tracks = {
            1: {
                "frame_ids": list(range(171, 601)),
                "positions": np.zeros((430, 2)),
                "bboxes": np.zeros((430, 4)),
            }
        }
        starts = [w["obs_start"] for w in
                  define_evaluation_windows(tracks, t_obs=60, t_pred=150)]
        assert starts == [151, 301]


# --------------------------------------------------------------------------- #
# 8. Submission format: window_start disambiguates colliding predictions
# --------------------------------------------------------------------------- #
class TestSubmissionFormat:
    """The 6-column form exists because windows overlap in time.

    Windows advance by t_pred while each predicts t_pred frames, so a frame can
    sit in the prediction horizon of more than one window. Two such predictions
    are both legitimate -- they were made from different observations -- and
    without window_start they share a key and one is silently lost.
    """

    def test_six_column_rows_are_keyed_by_window(self, tmp_path):
        f = tmp_path / "pred.txt"
        f.write_text(
            "1,61,7,0,100.0,200.0\n"      # window 1 predicts frame 61
            "151,61,7,0,999.0,888.0\n"    # window 151 also predicts frame 61
        )
        pred = _load_predictions(f)
        assert set(pred) == {1, 151}
        np.testing.assert_allclose(pred[1][7][0][61], [100.0, 200.0])
        np.testing.assert_allclose(pred[151][7][0][61], [999.0, 888.0])

    def test_legacy_rows_land_under_sentinel(self, tmp_path):
        f = tmp_path / "pred.txt"
        f.write_text("61,7,100.0,200.0,0\n")
        pred = _load_predictions(f)
        assert set(pred) == {-1}
        np.testing.assert_allclose(pred[-1][7][0][61], [100.0, 200.0])

    def test_legacy_form_cannot_separate_windows(self, tmp_path):
        """Why the column was added: the old form loses one of the two rows."""
        f = tmp_path / "pred.txt"
        f.write_text(
            "61,7,100.0,200.0,0\n"
            "61,7,999.0,888.0,0\n"
        )
        pred = _load_predictions(f)
        assert len(pred[-1][7][0]) == 1        # one survives, not two

    def test_modes_are_kept_apart(self, tmp_path):
        f = tmp_path / "pred.txt"
        f.write_text("".join(
            f"1,61,7,{k},{100.0 + k},200.0\n" for k in range(6)
        ))
        pred = _load_predictions(f)
        assert sorted(pred[1][7]) == list(range(6))

    def test_short_and_blank_lines_are_skipped(self, tmp_path):
        f = tmp_path / "pred.txt"
        f.write_text("\n1,61,7,0,100.0,200.0\n1,2,3\n")
        pred = _load_predictions(f)
        assert set(pred) == {1}

    def test_float_formatted_ids_are_accepted(self, tmp_path):
        """Numpy savetxt and pandas both emit integer columns as floats."""
        f = tmp_path / "pred.txt"
        f.write_text("1.0,61.0,7.0,0.0,100.0,200.0\n")
        pred = _load_predictions(f)
        np.testing.assert_allclose(pred[1][7][0][61], [100.0, 200.0])


# --------------------------------------------------------------------------- #
# 9. Association cutoff derives from the metric, not a constant
# --------------------------------------------------------------------------- #
class TestAssociationDistance:
    def test_cutoff_is_where_the_weakest_alpha_gives_up(self):
        """At the cutoff, DotD equals alpha_min exactly."""
        S = 24.15
        d = association_max_distance(S, alpha_min=0.05)
        np.testing.assert_allclose(np.exp(-d / S), 0.05)

    def test_cutoff_scales_with_the_dataset(self):
        """It is a multiple of S, so a different box size rescales it."""
        ratios = [association_max_distance(S) / S for S in (8.0, 24.15, 64.0)]
        np.testing.assert_allclose(ratios, ratios[0])

    def test_smot4sb_cutoff_is_about_three_boxes(self):
        assert association_max_distance(24.15) == pytest.approx(72.3, abs=0.1)

    def test_stricter_alpha_gives_a_tighter_cutoff(self):
        S = 24.15
        assert association_max_distance(S, 0.5) < association_max_distance(S, 0.05)


# --------------------------------------------------------------------------- #
# 10. Scoring a whole split (what the leaderboard runs)
# --------------------------------------------------------------------------- #
class TestEvaluateSplit:
    """A split score is not the mean of per-sequence scores.

    SO-HOTA weights association by true-positive count, as TrackEval's
    combine_sequences does, so a sequence with forty windows must outweigh one
    with two.
    """

    @staticmethod
    def _write_seq(d, seq, n_tracks=1, n_frames=400):
        """A GT sequence long enough to yield windows, plus a perfect prediction."""
        gt, pred, trk = [], [], []
        for tid in range(1, n_tracks + 1):
            for f in range(1, n_frames + 1):
                x, y = 100.0 + f + tid * 50, 200.0 + f
                gt.append(f"{f},{tid},{x - 8},{y - 8},16,16,1,-1,-1,-1")
        (d / "gt").mkdir(exist_ok=True)
        (d / "gt" / f"{seq}.txt").write_text("\n".join(gt) + "\n")

        for win in range(1, n_frames - 209 + 1, 150):
            for tid in range(1, n_tracks + 1):
                for f in range(win, win + 60):
                    x, y = 100.0 + f + tid * 50, 200.0 + f
                    trk.append(f"{f},{tid},{x - 8},{y - 8},16,16,1.0")
                for k in range(6):
                    for i in range(150):
                        f = win + 60 + i
                        x, y = 100.0 + f + tid * 50, 200.0 + f
                        pred.append(f"{win},{f},{tid},{k},{x},{y}")
        for sub, rows in (("prediction", pred), ("tracking", trk)):
            (d / sub).mkdir(exist_ok=True)
            (d / sub / f"{seq}.txt").write_text("\n".join(rows) + "\n")

    def test_perfect_predictions_score_one(self, tmp_path):
        self._write_seq(tmp_path, "0001")
        from evaluation.so_hota_tp import evaluate_split

        r = evaluate_split(
            tmp_path / "prediction", tmp_path / "gt", tmp_path / "tracking"
        )
        assert r["SO-HOTA-TP"] == pytest.approx(1.0, abs=1e-6)
        assert r["minADE"] == pytest.approx(0.0, abs=1e-6)

    def test_a_missing_sequence_scores_zero_not_skipped(self, tmp_path):
        """Otherwise omitting a hard sequence would raise the score."""
        self._write_seq(tmp_path, "0001")
        self._write_seq(tmp_path, "0002")
        from evaluation.so_hota_tp import evaluate_split

        full = evaluate_split(
            tmp_path / "prediction", tmp_path / "gt", tmp_path / "tracking"
        )
        (tmp_path / "prediction" / "0002.txt").unlink()
        partial = evaluate_split(
            tmp_path / "prediction", tmp_path / "gt", tmp_path / "tracking"
        )

        assert partial["n_missing_sequences"] == 1
        assert partial["per_sequence"]["0002"]["missing"] is True
        assert partial["SO-HOTA-TP"] < full["SO-HOTA-TP"]

    def test_S_is_shared_across_sequences(self, tmp_path):
        """One DotD scale for the split, so boxes mean the same thing in each."""
        self._write_seq(tmp_path, "0001")
        self._write_seq(tmp_path, "0002")
        from evaluation.so_hota_tp import evaluate_split

        r = evaluate_split(
            tmp_path / "prediction", tmp_path / "gt", tmp_path / "tracking"
        )
        assert r["S"] == pytest.approx(16.0)   # every box is 16x16

    def test_empty_gt_directory_raises(self, tmp_path):
        (tmp_path / "gt").mkdir()
        (tmp_path / "prediction").mkdir()
        from evaluation.so_hota_tp import evaluate_split

        with pytest.raises(ValueError, match="no ground-truth"):
            evaluate_split(tmp_path / "prediction", tmp_path / "gt")
