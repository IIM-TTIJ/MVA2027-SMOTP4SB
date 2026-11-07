"""prediction.paths: moving generated data must not lose the shipped split."""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _reload(monkeypatch, data_dir):
    monkeypatch.setenv("SMOTP4SB_DATA", str(data_dir))
    import prediction.paths as paths
    return importlib.reload(paths)


def test_split_falls_back_to_the_shipped_file(monkeypatch, tmp_path):
    paths = _reload(monkeypatch, tmp_path)
    assert paths.SPLIT == ROOT / "data" / "traj" / "split.json"
    assert paths.SPLIT.is_file()


def test_a_generated_split_takes_precedence(monkeypatch, tmp_path):
    (tmp_path / "traj").mkdir()
    (tmp_path / "traj" / "split.json").write_text("{}")
    paths = _reload(monkeypatch, tmp_path)
    assert paths.SPLIT == tmp_path / "traj" / "split.json"


def teardown_module():
    os.environ.pop("SMOTP4SB_DATA", None)
    import prediction.paths as paths
    importlib.reload(paths)
