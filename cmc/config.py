"""Single place for CMC configuration.

Where the camera intrinsics and rotation files are kept, and the master on/off
switch. Nothing else in the codebase should hardcode any of it.

Intrinsics are not a config value: K varies per sequence, so this only records
which file to read and which split to resolve ids against. See cmc.intrinsics.
"""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from prediction import paths

FPS = 30.0


@dataclass
class CMCConfig:
    """Runtime CMC settings.

    enabled=False must leave the uncompensated baseline bit-identical: every
    call site is gated on it, and the identity test in tests/test_cmc.py exists
    to prove the gate holds.
    """

    enabled: bool = False
    # Where per-frame camera motion comes from.
    #   "egomotion"  imu.csv through the fitted IMU -> image-motion model
    #                (cmc/egomotion.py). The default: the dataset's camera is on
    #                a pilot-steered gimbal with an optical zoom, so a rotation
    #                homography from the airframe attitude does not describe it.
    #   "rotations"  (T, 3, 3) world-to-camera rotations from rotation_dir and
    #                K from camera_intrinsics.json (synthetic data, tests).
    source: str = "egomotion"
    egomotion_model: Path = field(
        default_factory=lambda: paths.ROOT / "cmc" / "egomotion_model.json")
    # Directory holding {split}/{sequence}/imu.csv; None means the dataset
    # itself. $SMOTP4SB_IMU_DIR sets it without a config file.
    imu_dir: Path | None = field(default_factory=lambda: (
        Path(os.environ["SMOTP4SB_IMU_DIR"]) if os.environ.get("SMOTP4SB_IMU_DIR") else None))
    # Camera motion assumed over the prediction horizon under mode="full",
    # where future IMU must not be used: "hold" keeps the last observed frame.
    future: str = "hold"
    intrinsics: Path = field(
        default_factory=lambda: paths.SMOT4SB / "camera_intrinsics.json")
    # Which split a bare sequence id belongs to. Intrinsics are keyed
    # "{split}/{sequence}" because ids repeat across splits -- train/0001 and
    # pub_test/0001 are different footage. The Track 2 pipeline reads
    # annotations/train.json only, so "train" is right until test-split scoring
    # exists; it is a field rather than a constant so that day needs no edit here.
    split: str = "train"
    rotation_dir: Path = field(default_factory=lambda: paths.DATA / "rotations")
    # Which part of the window is compensated.
    #   "full"       observation and target both, so the model predicts in
    #                reference-frame coordinates and the prediction must be
    #                mapped back to raw pixels before scoring.
    #   "input_only" observation only. The model sees de-rotated history but
    #                predicts in raw pixels, so no future rotation is forecast
    #                and none of the extrapolation error applies. The cost is
    #                that the model must absorb the camera's motion over the
    #                prediction window implicitly.
    mode: str = "full"
    extrapolation: str = "constant_omega"
    # Which observed frame the window is expressed in. The default is the
    # first. "last" leaves 60 frames less accumulated rotation between the
    # reference and the predicted frames, which matters because projective
    # error amplification grows with that gap -- see docs/cmc.md.
    reference: str = "first"
    fps: float = FPS

    @classmethod
    def load(cls, path=None) -> "CMCConfig":
        """Read a JSON config, falling back to defaults for anything absent.

        Order of precedence: explicit path > $SMOTP4SB_CMC_CONFIG > defaults.
        """
        path = path or os.environ.get("SMOTP4SB_CMC_CONFIG")
        cfg = cls()
        if not path:
            return cfg
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"CMC config must be a JSON object, got {type(raw).__name__}")
        if "intrinsics" in raw:
            raw["intrinsics"] = Path(raw["intrinsics"])
        for key in ("rotation_dir", "egomotion_model", "imu_dir"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        unknown = set(raw) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"unknown CMC config keys: {sorted(unknown)}")
        return replace(cfg, **raw)

    def with_mode(self, mode: str) -> "CMCConfig":
        """Switch compensation mode, moving the reference frame with it.

        input_only is only defined against the last observed frame -- see
        validate() -- so the two always travel together. Asking every caller to
        remember both is how they end up out of step.
        """
        return replace(self, mode=mode,
                       reference="last" if mode == "input_only" else self.reference)

    def validate(self) -> "CMCConfig":
        # K itself is validated in cmc.intrinsics as each sequence is read,
        # since there is no single K to check here.
        if not self.split or "/" in self.split:
            raise ValueError(f"split must be a bare split name, got {self.split!r}")
        if self.mode not in ("full", "input_only"):
            raise ValueError(f"mode must be 'full' or 'input_only', got {self.mode!r}")
        if self.reference not in ("first", "last"):
            raise ValueError(f"reference must be 'first' or 'last', got {self.reference!r}")
        # input_only is defined against the last observed frame: there the
        # homography is exactly the identity, so the anchor keeps its true pixel
        # position and the raw target stays consistent with it. Referencing the
        # first frame would move the anchor while the target stayed raw, leaving
        # a per-window offset for the model to guess. Rejecting the combination
        # beats silently ignoring cfg.reference.
        if self.mode == "input_only" and self.reference != "last":
            raise ValueError(
                "mode='input_only' requires reference='last'; "
                f"got reference={self.reference!r}")
        if self.source not in ("egomotion", "rotations"):
            raise ValueError(f"source must be 'egomotion' or 'rotations', got {self.source!r}")
        if self.future != "hold":
            raise ValueError(f"future must be 'hold', got {self.future!r}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.extrapolation not in ("constant_omega", "uniform", "oracle"):
            raise ValueError(f"unknown extrapolation mode {self.extrapolation!r}")
        return self
