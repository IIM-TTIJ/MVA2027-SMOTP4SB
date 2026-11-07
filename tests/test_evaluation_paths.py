"""Do the evaluation scripts run against the new dataset paths?

The 2027 drop moved the data into a track-based layout, so the question this
answers is narrow: point each evaluation script at a new root and check it
still runs and still produces metrics.

Each script is launched as a real subprocess, the way it is run by hand, rather
than by importing its internals.

Skips when no root is reachable, since an unmounted share is an environment
problem and not a code defect.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prediction import paths                                    # noqa: E402

# The evaluation entry points that read the dataset root. eval_prediction.py is
# absent on purpose: it takes --pred and --gt file paths, so the move cannot
# affect it. eval_tracking.py needs Track 1 tracker output, which is not here.
EVAL_SCRIPTS = (
    "prediction.eval_constant_velocity",
    "prediction.motion_models",
    "prediction.kalman",
)

# Extra dataset roots to check, os.pathsep-separated, for example both track
# packages side by side. Each root must hold annotations/train.json.
NEW_ROOTS = [Path(p) for p in os.environ.get("SMOT4SB_TEST_ROOTS", "").split(os.pathsep) if p]

ROOTS = [r for r in NEW_ROOTS if (r / "annotations" / "train.json").is_file()]
if not ROOTS and (paths.ANNOTATIONS).is_file():
    ROOTS = [paths.SMOT4SB]                     # fall back to whatever is configured

METRIC = re.compile(r"\d+\.\d{2}")


def _label(v):
    """Ids read as tracking-public / kalman, not full paths."""
    if isinstance(v, Path):
        return f"{v.parent.name.replace('SMOT4SB_', '')}-{v.name}"
    return str(v).rsplit(".", 1)[-1]


@pytest.mark.skipif(not ROOTS, reason="no dataset root reachable")
@pytest.mark.parametrize("root", ROOTS, ids=_label)
@pytest.mark.parametrize("module", EVAL_SCRIPTS, ids=_label)
def test_evaluation_script_runs_against_the_new_path(module, root):
    env = dict(os.environ, SMOT4SB_ROOT=str(root), PYTHONPATH=str(ROOT))
    p = subprocess.run([sys.executable, "-m", module],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)

    assert p.returncode == 0, f"{module} failed on {root}\n{p.stderr[-2000:]}"

    values = METRIC.findall(p.stdout)
    assert values, f"{module} printed no metrics:\n{p.stdout[-1000:]}"
    # A script that evaluated zero windows would still exit 0, so require the
    # output to carry a real figure rather than only structure.
    assert any(float(v) > 0 for v in values), f"{module} reported only zeros"
    assert "nan" not in p.stdout.lower(), f"{module} produced NaN:\n{p.stdout[-1000:]}"
