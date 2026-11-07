"""tools/validate_submission.py: what blocks an upload and what does not.

The scorer judges every predicted frame on its own, so a track may be absent
for any subset of the horizon: it may stop (the bird leaves), start late or
resume after a gap (the bird was occluded). Only shapes the scorer cannot read
are errors.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from validate_submission import T_OBS, T_PRED, K, validate  # noqa: E402

WIN = 1
LO, HI = WIN + T_OBS, WIN + T_OBS + T_PRED - 1


def _write(tmp: Path, frames: list[int]) -> Path:
    (tmp / "prediction").mkdir()
    (tmp / "tracking").mkdir()
    rows = [f"{WIN},{f},7,{k},100.0,200.0" for f in frames for k in range(K)]
    (tmp / "prediction" / "0001.txt").write_text("\n".join(rows) + "\n")
    obs = [f"{f},7,92.0,192.0,16.0,16.0,1.0" for f in range(WIN, WIN + T_OBS)]
    (tmp / "tracking" / "0001.txt").write_text("\n".join(obs) + "\n")
    return tmp


def test_full_horizon_passes(tmp_path):
    assert validate(_write(tmp_path, list(range(LO, HI + 1))))


def test_stopping_early_passes(tmp_path):
    assert validate(_write(tmp_path, list(range(LO, LO + 40))))


def test_resuming_after_a_gap_passes(tmp_path):
    frames = list(range(LO, LO + 30)) + list(range(LO + 60, HI + 1))
    assert validate(_write(tmp_path, frames))


def test_starting_late_passes(tmp_path):
    assert validate(_write(tmp_path, list(range(LO + 20, HI + 1))))


def test_frames_outside_the_horizon_fail(tmp_path):
    assert not validate(_write(tmp_path, list(range(LO - 5, LO + 10))))
