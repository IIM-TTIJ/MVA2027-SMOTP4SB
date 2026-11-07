"""Tests for prediction/infer.py -- turning tracking output into submissions."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prediction.infer import (
    infer_sequence,
    load_tracks,
    predict_cv,
    predict_stay,
    windows_for,
)


def _mot(path: Path, n_tracks=1, n_frames=400):
    rows = []
    for tid in range(1, n_tracks + 1):
        for f in range(1, n_frames + 1):
            x, y = 100.0 + f + tid * 50, 200.0 + f
            rows.append(f"{f},{tid},{x - 8},{y - 8},16,16,1.0")
    path.write_text("\n".join(rows) + "\n")


class TestLoadTracks:
    def test_boxes_become_centres(self, tmp_path):
        f = tmp_path / "s.txt"
        f.write_text("1,7,100,200,20,40,1.0\n")
        np.testing.assert_allclose(load_tracks(f)[7][1], [110.0, 220.0])

    def test_blank_and_short_lines_are_skipped(self, tmp_path):
        f = tmp_path / "s.txt"
        f.write_text("\n1,7,100,200,20,40,1.0\n1,2,3\n")
        assert list(load_tracks(f)) == [7]


class TestWindows:
    def test_windows_match_the_scorer(self, tmp_path):
        """Windows must line up with evaluation.so_hota_tp, or predictions are
        keyed to windows the leaderboard never evaluates."""
        from evaluation.so_hota_tp import define_evaluation_windows

        f = tmp_path / "s.txt"
        _mot(f, n_frames=600)
        tracks = load_tracks(f)
        mine = windows_for(tracks, 60, 150)

        gt = {tid: {"frame_ids": sorted(fr), "positions": np.zeros((len(fr), 2)),
                    "bboxes": np.zeros((len(fr), 4))}
              for tid, fr in tracks.items()}
        theirs = [w["obs_start"] for w in define_evaluation_windows(gt, 60, 150)]
        assert mine == theirs

    def test_too_short_yields_nothing(self, tmp_path):
        f = tmp_path / "s.txt"
        _mot(f, n_frames=100)          # < 60 + 150
        assert windows_for(load_tracks(f), 60, 150) == []


class TestSimplePredictors:
    def test_stay_repeats_the_last_point(self):
        obs = np.arange(120, dtype=float).reshape(60, 2)
        out = predict_stay(obs, 150, 6)
        assert out.shape == (6, 150, 2)
        np.testing.assert_allclose(out, np.broadcast_to(obs[-1], out.shape))

    def test_cv_extrapolates_the_last_step(self):
        obs = np.stack([np.arange(60.0), np.zeros(60)], axis=1)
        out = predict_cv(obs, 150, 6)
        np.testing.assert_allclose(out[0, 0], [60.0, 0.0])
        np.testing.assert_allclose(out[0, -1], [209.0, 0.0])

    def test_deterministic_modes_are_identical(self):
        """So minADE_K collapses to ADE_1 for these, by construction."""
        obs = np.arange(120, dtype=float).reshape(60, 2)
        for fn in (predict_stay, predict_cv):
            out = fn(obs, 150, 6)
            for k in range(1, 6):
                np.testing.assert_allclose(out[0], out[k])


class TestInferSequence:
    def test_row_count_and_shape(self, tmp_path):
        f = tmp_path / "s.txt"
        _mot(f, n_tracks=2, n_frames=400)
        tracks = load_tracks(f)
        rows = infer_sequence(
            tracks, lambda o: predict_stay(o, 150, 6), False, 60, 150, 6)

        n_win = len(windows_for(tracks, 60, 150))
        assert len(rows) == n_win * 2 * 6 * 150
        win, fid, tid, k, cx, cy = rows[0].split(",")
        assert int(fid) == int(win) + 60      # first predicted frame
        assert 0 <= int(k) < 6

    def test_output_parses_as_a_submission(self, tmp_path):
        """The rows must round-trip through the scorer's own reader."""
        from evaluation.so_hota_tp import _load_predictions

        f = tmp_path / "s.txt"
        _mot(f, n_frames=400)
        rows = infer_sequence(
            load_tracks(f), lambda o: predict_stay(o, 150, 6), False, 60, 150, 6)
        out = tmp_path / "pred.txt"
        out.write_text("\n".join(rows) + "\n")

        pred = _load_predictions(out)
        assert -1 not in pred                  # keyed by window, not the legacy form
        for win, by_track in pred.items():
            for tid, modes in by_track.items():
                assert sorted(modes) == list(range(6))
                for frames in modes.values():
                    assert len(frames) == 150

    def test_tracks_seen_too_briefly_are_dropped(self, tmp_path):
        """Under 5 observed frames the scorer cannot associate the track."""
        f = tmp_path / "s.txt"
        rows = [f"{i},1,100,200,16,16,1.0" for i in range(1, 401)]
        rows += [f"{i},2,500,600,16,16,1.0" for i in range(1, 4)]  # 3 frames
        f.write_text("\n".join(rows) + "\n")

        out = infer_sequence(
            load_tracks(f), lambda o: predict_stay(o, 150, 6), False, 60, 150, 6)
        assert {r.split(",")[2] for r in out} == {"1"}

    def test_short_observations_are_left_padded(self, tmp_path):
        """A gappy track still yields a full-length window for the model."""
        f = tmp_path / "s.txt"
        rows = [f"{i},1,{100 + i},200,16,16,1.0" for i in range(1, 401) if i % 2]
        f.write_text("\n".join(rows) + "\n")

        seen = []
        def spy(o):
            seen.append(o.shape)
            return predict_stay(o, 150, 6)

        infer_sequence(load_tracks(f), spy, False, 60, 150, 6)
        assert seen and all(s == (60, 2) for s in seen)

    def test_compensation_reaches_the_model(self, tmp_path):
        """A --cmc checkpoint must see the observation it was trained on."""
        f = tmp_path / "s.txt"
        _mot(f, n_frames=400)
        shift = lambda obs, start: obs + np.array([10.0, 0.0])
        plain = infer_sequence(
            load_tracks(f), lambda o: predict_stay(o, 150, 6), False, 60, 150, 6)
        moved = infer_sequence(
            load_tracks(f), lambda o: predict_stay(o, 150, 6), False, 60, 150, 6,
            compensate=shift)
        assert len(plain) == len(moved)
        for a, b in zip(plain, moved):
            assert float(b.split(",")[4]) == pytest.approx(float(a.split(",")[4]) + 10)

    def test_compensation_is_aligned_to_frames(self, tmp_path):
        """Each observed point is compensated with its own frame's row; unseen
        frames reach the compensator as NaN and never the model."""
        f = tmp_path / "s.txt"
        rows = [f"{i},1,{100 + i},200,16,16,1.0" for i in range(1, 401) if i % 2]
        f.write_text("\n".join(rows) + "\n")

        calls = []
        def compensate(obs, start):
            calls.append((start, obs.copy()))
            return obs
        seen = []
        def spy(o):
            seen.append(o)
            return predict_stay(o, 150, 6)

        infer_sequence(load_tracks(f), spy, False, 60, 150, 6, compensate=compensate)
        start, obs = calls[0]
        assert start == 1 and obs.shape == (60, 2)
        assert np.isnan(obs[1::2]).all()                     # frames 2, 4, ... unseen
        np.testing.assert_allclose(obs[::2, 0], 108 + np.arange(1, 61, 2))   # box centres
        assert all(np.isfinite(o).all() for o in seen)


class TestWindowAlignment:
    """Windows are anchored to the sequence, not to what the tracker found.

    A detector that misses the opening frames would otherwise shift every
    window, and every prediction would be keyed to a window the scorer never
    cuts -- scoring zero while the file looks perfectly well-formed. This was a
    real bug: real tracker output first fired at frame 53, producing windows at
    53/203/353 against the scorer's 1/151/301, and nothing matched.
    """

    def test_late_first_detection_does_not_shift_windows(self, tmp_path):
        f = tmp_path / "s.txt"
        # Nothing detected until frame 53, as a real detector behaves.
        rows = [f"{i},1,{100 + i},200,16,16,1.0" for i in range(53, 601)]
        f.write_text("\n".join(rows) + "\n")

        wins = windows_for(load_tracks(f), 60, 150)
        assert wins[0] == 1, "windows must start at the sequence's first frame"

    def test_matches_the_scorer_on_late_starting_tracks(self, tmp_path):
        from evaluation.so_hota_tp import define_evaluation_windows

        f = tmp_path / "s.txt"
        rows = [f"{i},1,{100 + i},200,16,16,1.0" for i in range(53, 601)]
        f.write_text("\n".join(rows) + "\n")

        # Neither side may key windows to its own first box: the tracker starts
        # at 53, the GT at 79, and both grids must still be 1, 151, 301, ...
        gt = {1: {"frame_ids": list(range(79, 601)),
                  "positions": np.zeros((522, 2)),
                  "bboxes": np.zeros((522, 4))}}
        theirs = [w["obs_start"] for w in define_evaluation_windows(gt, 60, 150)]
        mine = windows_for(load_tracks(f), 60, 150)
        assert theirs and set(theirs) <= set(mine)
