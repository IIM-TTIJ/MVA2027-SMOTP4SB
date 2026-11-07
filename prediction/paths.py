"""Filesystem locations, resolved against the repository instead of the CWD.

Every path here is derived from this file's own location, so the modules run
from any working directory rather than only from the repo root. Two locations
that may reasonably live elsewhere -- a shared dataset mount, a scratch disk --
can be pointed somewhere else with environment variables:

    SMOT4SB_ROOT     the SMOT4SB dataset      (default: <repo>/SMOT4SB)
    SMOTP4SB_DATA   generated data artifacts (default: <repo>/data)

CLI arguments still win over both: a relative --split is taken from the repo
root, an absolute one is used as given.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SMOT4SB = Path(os.environ.get("SMOT4SB_ROOT") or ROOT / "SMOT4SB")
ANNOTATIONS = SMOT4SB / "annotations" / "train.json"

DATA = Path(os.environ.get("SMOTP4SB_DATA") or ROOT / "data")
TRAJ = DATA / "traj"
# The validation split ships with the repository. A split generated into DATA
# wins, but moving DATA elsewhere must not make the shipped one disappear.
SHIPPED_SPLIT = ROOT / "data" / "traj" / "split.json"
SPLIT = TRAJ / "split.json" if (TRAJ / "split.json").is_file() else SHIPPED_SPLIT

PPT = ROOT / "PPT"
PPT_DATA = PPT / "data"
RUNS = ROOT / "runs"


def resolve(p) -> Path:
    """Absolute paths pass through; relative ones are taken from the repo root.

    Without this a relative default like "data/traj/split.json" silently means
    something different depending on where the command was launched from.
    """
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def require(path: Path, what: str, hint: str) -> Path:
    """Fail with an actionable message instead of a bare FileNotFoundError."""
    if not path.exists():
        raise SystemExit(f"{what} not found at {path}\n  {hint}")
    return path
