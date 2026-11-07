#!/bin/bash
# Install the OpenMMLab stack (mmengine / mmcv / mmdet / mmyolo) at the versions
# this repo is known to work on. Track 1 only -- Track 2 (trajectory prediction)
# needs none of this; see README.md.
#
# The one thing that matters here is that **mmcv must come from OpenMMLab's
# prebuilt wheel index**, not from `pip install mmcv` or `mim install mmcv`.
# Those build the CUDA extension from source, which takes tens of minutes and
# usually fails on a CUDA/compiler mismatch. The prebuilt index is keyed by
# (CUDA version, torch version) and ships mmcv._ext already compiled.
#
# Version constraints that pin the choice:
#   mmdet 3.3.0   requires mmcv >=2.0.0rc4, <2.2.0   <-- the binding cap
#   mmyolo 0.5.0  requires mmcv >=2.0.0rc4, <2.3.0
#                 requires mmdet >=3.0.0rc6, <4.0.0
# So mmcv 2.1.0 is the newest version mmdet will accept.
#
# NOTE: this script was reconstructed from a working install (the original
# machine-specific setup script was never committed). The versions below are
# read off that environment and verified; the wheel index URL follows
# OpenMMLab's standard layout.
#
# Usage:
#   scripts/setup_openmmlab.sh            # into the active venv
set -euo pipefail

PY_TAG=cp310                 # Python 3.10
TORCH=2.1.0
TORCHVISION=0.16.0
CUDA=cu121
MMCV=2.1.0
MMENGINE=0.10.7
MMDET=3.3.0

PIP=${PIP:-uv pip}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> torch ${TORCH}+${CUDA}"
$PIP install "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "==> mmengine ${MMENGINE}"
$PIP install "mmengine==${MMENGINE}"

# The whole point of this script. Do NOT replace with `mim install mmcv`.
echo "==> mmcv ${MMCV} (prebuilt wheel, ${CUDA}/torch${TORCH})"
$PIP install "mmcv==${MMCV}" \
    -f "https://download.openmmlab.com/mmcv/dist/${CUDA}/torch${TORCH}/index.html"

echo "==> mmdet ${MMDET}"
$PIP install "mmdet==${MMDET}"

echo "==> mmyolo (editable, from the YOLOv8-SMOT submodule)"
if [ -d "$ROOT/YOLOv8-SMOT/mmyolo" ]; then
    $PIP install -e "$ROOT/YOLOv8-SMOT/mmyolo"
else
    echo "    SKIP: YOLOv8-SMOT/mmyolo not found. Run:"
    echo "      git submodule update --init --recursive"
fi

echo
echo "==> verifying"
python - <<'PY'
import sys
import torch, mmcv, mmengine, mmdet
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}  (CUDA {torch.version.cuda}, available={torch.cuda.is_available()})")
print(f"  mmengine    {mmengine.__version__}")
print(f"  mmcv        {mmcv.__version__}")
print(f"  mmdet       {mmdet.__version__}  needs mmcv [{mmdet.mmcv_minimum_version}, {mmdet.mmcv_maximum_version})")
try:
    import mmyolo
    print(f"  mmyolo      {mmyolo.__version__}  needs mmcv [{mmyolo.mmcv_minimum_version}, {mmyolo.mmcv_maximum_version})")
except ImportError:
    print("  mmyolo      NOT INSTALLED")
# The check that actually matters: a pure-python mmcv imports fine but has no
# ops, and every detector call fails later with a confusing error.
from mmcv.ops import nms          # noqa: F401
print("  mmcv.ops    OK -- compiled CUDA extension present")
PY
echo
echo "Done."
