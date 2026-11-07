import tempfile
from pathlib import Path

import numpy as np
import pytest

from prediction.data_utils import load_mot_tracks, extract_windows


def _write_mot(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


class TestLoadMotTracks:
    def test_basic_loading(self, tmp_path: Path) -> None:
        mot_file = tmp_path / "tracks.txt"
        _write_mot(mot_file, [
            "1,1,100.0,200.0,20.0,15.0,1,1,1",
            "2,1,105.0,202.0,20.0,15.0,1,1,1",
            "1,2,300.0,400.0,10.0,10.0,1,1,1",
        ])
        tracks = load_mot_tracks(mot_file)
        assert set(tracks.keys()) == {1, 2}
        assert tracks[1].shape == (2, 2)  # 2 frames, (cx, cy)
        np.testing.assert_allclose(tracks[1][0], [110.0, 207.5])  # cx = x + w/2
        assert tracks[2].shape == (1, 2)

    def test_empty_file(self, tmp_path: Path) -> None:
        mot_file = tmp_path / "empty.txt"
        mot_file.write_text("")
        tracks = load_mot_tracks(mot_file)
        assert tracks == {}

    def test_frame_ids_preserved(self, tmp_path: Path) -> None:
        mot_file = tmp_path / "tracks.txt"
        _write_mot(mot_file, [
            "5,1,10.0,20.0,4.0,4.0,1,1,1",
            "8,1,14.0,24.0,4.0,4.0,1,1,1",
        ])
        tracks = load_mot_tracks(mot_file, return_frame_ids=True)
        assert tracks[1]["frame_ids"] == [5, 8]


class TestExtractWindows:
    def test_basic_windowing(self) -> None:
        positions = np.array([[i * 1.0, i * 2.0] for i in range(10)])
        frame_ids = list(range(1, 11))
        windows = extract_windows(positions, frame_ids, t_obs=3, t_pred=3, stride=1)
        assert len(windows) > 0
        obs, pred, meta = windows[0]
        assert obs.shape == (3, 2)
        assert pred.shape == (3, 2)

    def test_track_too_short(self) -> None:
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        frame_ids = [1, 2]
        windows = extract_windows(positions, frame_ids, t_obs=3, t_pred=3, stride=1)
        assert len(windows) == 0

    def test_gap_handling(self) -> None:
        positions = np.array([[i * 1.0, i * 1.0] for i in range(6)])
        frame_ids = [1, 2, 3, 10, 11, 12]  # gap between 3 and 10
        windows = extract_windows(positions, frame_ids, t_obs=3, t_pred=3, stride=1)
        assert len(windows) == 0  # gap breaks continuity
