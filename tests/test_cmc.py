"""Tests for camera motion compensation, in the order the plan specifies.

1. identity   -- CMC enabled with a still camera must change nothing at all.
2. round-trip -- compensate then undo must return the original pixels.
3. smoke      -- the pipeline must survive a real train+eval with CMC on.

The identity test is the one that catches plumbing bugs: if enabling CMC with
identity rotations moves any number, something is being applied twice, applied
to the wrong axis, or applied where it should not be.
"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cmc import pipeline                                          # noqa: E402
from cmc.compensate import compensate_trajectory, inverse_compensate  # noqa: E402
from cmc.config import CMCConfig                                  # noqa: E402
from cmc.extrapolate import extrapolate                           # noqa: E402
from cmc.intrinsics import K_for                                  # noqa: E402
from cmc.rotations import (check_rotations, from_euler, from_quaternion,  # noqa: E402
                           save_rotations, synthetic)
from prediction import paths                                      # noqa: E402

T_OBS, T_PRED = 60, 150
WIN = T_OBS + T_PRED

# Real matrices from the dataset, not invented ones: siyi_a8mini at 4K and
# mavic2pro at 1080p, the two ends of the focal range (2248 and 1492).
K_4K = np.array([[2248.0,    0.0, 1920.0],
                 [   0.0, 2248.0, 1080.0],
                 [   0.0,    0.0,    1.0]])
K_FHD = np.array([[1492.0,   0.0, 960.0],
                  [  0.0, 1492.0, 540.0],
                  [  0.0,    0.0,   1.0]])

# The synthetic videos the tests below invent rotations for.
FAKE_SEQUENCES = ("id01", "rt01", "or01", "short", "bit01", "rt_last", "obs01",
                  "obsshort", "stillfull")


def write_intrinsics(path, entries=None):
    """Write a camera_intrinsics.json covering the tests' synthetic video ids.

    The tests stay runnable without the dataset mounted while still going
    through the real lookup rather than a stub.
    """
    if entries is None:
        entries = {f"train/{s}": (K_4K, (3840, 2160)) for s in FAKE_SEQUENCES}
    path.write_text(json.dumps({
        "description": "test fixture",
        "cameras": {},
        "sequences": {k: {"camera": "siyi_a8mini", "resolution": list(res),
                          "K": K.tolist(), "has_imu": True}
                      for k, (K, res) in entries.items()},
    }))
    return path


@pytest.fixture
def cfg(tmp_path):
    return CMCConfig(enabled=True, source="rotations", rotation_dir=tmp_path,
                     intrinsics=write_intrinsics(tmp_path / "ci.json")).validate()


@pytest.fixture
def traj():
    rng = np.random.default_rng(0)
    return rng.uniform([200.0, 200.0], [3600.0, 1900.0], size=(WIN, 2))


# --------------------------------------------------------------------------- #
# 1. Identity
# --------------------------------------------------------------------------- #
def test_identity_rotations_leave_pixels_untouched(traj):
    """A still camera must leave every pixel where it was.

    This used to assert bit-identity, which held only because the placeholder
    K had power-of-two-friendly entries: 1920/3840 and 1080/3840 are exact
    binary fractions, so K @ inv(K) came out as exactly I. Of the four real
    matrices only siyi_a8mini's breaks that -- 1920/2248 does not terminate in
    binary -- and it does so by 5e-13 px, eleven orders below the 48 px miss
    threshold. The tolerance here is still nine orders tighter than any
    plumbing bug this test exists to catch, all of which move whole pixels.
    """
    R = synthetic(WIN, mode="identity")
    out = compensate_trajectory(traj, R, K_4K)
    assert np.abs(out - traj).max() < 1e-9


def test_identity_through_the_pipeline_is_a_noop(traj, cfg):
    """Same guarantee one level up, where the frame bookkeeping lives."""
    save_rotations("id01", synthetic(WIN + 10, mode="identity"), cfg)
    comp = pipeline.compensate_window(traj, "id01", start_frame=1, cfg=cfg)
    assert np.allclose(comp, traj, atol=1e-9)

    modes = np.stack([traj[T_OBS:]] * 6)
    back = pipeline.decompensate_modes(modes, "id01", 1, T_OBS, cfg)
    assert np.allclose(back, modes, atol=1e-9)


def test_disabled_config_returns_input_unchanged(traj, tmp_path):
    """cmc.enabled=false must not even look at the rotation files."""
    off = CMCConfig(enabled=False, rotation_dir=tmp_path / "does-not-exist")
    assert pipeline.compensate_window(traj, "missing", 1, off) is traj
    modes = np.stack([traj[T_OBS:]] * 6)
    assert pipeline.decompensate_modes(modes, "missing", 1, T_OBS, off) is modes


# --------------------------------------------------------------------------- #
# 2. Round trip
# --------------------------------------------------------------------------- #
def test_round_trip_under_synthetic_rotation(traj):
    """compensate -> inverse_compensate must be the identity to < 1e-6 px."""
    R = synthetic(WIN)
    comp = compensate_trajectory(traj, R, K_4K)
    assert np.abs(comp - traj).max() > 1.0, "test is vacuous if nothing moved"
    back = inverse_compensate(comp, R, K_4K, R_ref=R[0])
    assert np.abs(back - traj).max() < 1e-6


def test_round_trip_for_all_k_modes(traj, cfg):
    """The K=6 mode stack must round-trip as a batch, not just one at a time."""
    save_rotations("rt01", synthetic(WIN + 10), cfg)
    R = pipeline.rotations_for("rt01", cfg)
    modes = np.stack([traj[T_OBS:] + i * 3.0 for i in range(6)])
    comp = compensate_trajectory(modes, R[T_OBS:WIN], K_4K)
    back = inverse_compensate(comp, R[T_OBS:WIN], K_4K, R_ref=R[T_OBS])
    assert np.abs(back - modes).max() < 1e-6


def test_oracle_extrapolation_round_trips_exactly(traj, cfg):
    """With true future rotations the whole pipeline is lossless.

    This isolates the CMC maths from the extrapolation: any error left here is
    a bug, whereas error under constant_omega is a modelling limitation.
    """
    save_rotations("or01", synthetic(WIN + 10), cfg)
    oracle_cfg = CMCConfig(enabled=True, source="rotations", rotation_dir=cfg.rotation_dir,
                           intrinsics=cfg.intrinsics, extrapolation="oracle")
    comp = pipeline.compensate_window(traj, "or01", 1, oracle_cfg)
    modes = np.stack([comp[T_OBS:]] * 6)
    tf = pipeline.true_future_rotations("or01", 1, T_OBS, T_PRED, oracle_cfg)
    back = pipeline.decompensate_modes(modes, "or01", 1, T_OBS, oracle_cfg,
                                       true_future=tf)
    assert np.abs(back - traj[T_OBS:]).max() < 1e-6


# --------------------------------------------------------------------------- #
# Supporting guarantees
# --------------------------------------------------------------------------- #
def test_rotation_validation_rejects_bad_input():
    with pytest.raises(ValueError, match="orthonormal"):
        check_rotations(np.tile(np.diag([1.0, 1.0, 2.0]), (3, 1, 1)))
    with pytest.raises(ValueError, match=r"det=\+1"):
        check_rotations(np.tile(np.diag([1.0, 1.0, -1.0]), (3, 1, 1)))
    with pytest.raises(ValueError, match=r"\(T, 3, 3\)"):
        check_rotations(np.eye(3))


def test_quaternion_convention_is_explicit():
    """scalar-first and scalar-last must give the same rotation for one input."""
    from scipy.spatial.transform import Rotation
    q_last = Rotation.random(8, random_state=1).as_quat()          # (x,y,z,w)
    q_first = q_last[:, [3, 0, 1, 2]]                              # (w,x,y,z)
    assert np.allclose(from_quaternion(q_last),
                       from_quaternion(q_first, scalar_first=True))


def test_euler_converter_matches_manual_rotation():
    R = from_euler(np.array([[90.0, 0.0, 0.0]]), seq="ZYX", degrees=True)[0]
    expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(R, expected, atol=1e-12)


def test_extrapolation_of_a_still_camera_is_identity():
    R = extrapolate(synthetic(T_OBS, mode="identity"), T_PRED)
    assert np.allclose(R, np.eye(3))


def test_reference_frame_mismatch_is_rejected():
    with pytest.raises(ValueError, match="R_ref"):
        inverse_compensate(np.zeros((6, T_PRED, 2)), synthetic(T_PRED),
                           K_4K, R_ref=synthetic(2))


def test_frame_range_error_names_the_video(cfg):
    save_rotations("short", synthetic(10), cfg)
    with pytest.raises(ValueError, match="short"):
        pipeline.compensate_window(np.zeros((WIN, 2)), "short", 1, cfg)


# --------------------------------------------------------------------------- #
# 3. End-to-end smoke
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_end_to_end_smoke_with_synthetic_rotations(tmp_path):
    """A real training run with CMC on must complete. The score is meaningless
    with fake rotations; only "it did not crash" is being asserted."""
    if not (ROOT / "data" / "traj" / "split.json").exists():
        pytest.skip("run prediction.build_dataset first")
    if not (paths.SMOT4SB / "annotations" / "train.json").is_file():
        pytest.skip("dataset not mounted")   # split.json ships with the repo; the data does not
    out = subprocess.run(
        [sys.executable, "-m", "prediction.learned_models",
         "--model", "lstm", "--epochs", "2", "--stride", "30"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "BEST" in out.stdout


# --------------------------------------------------------------------------- #
# Identity at eval-pipeline level
#
# Running the real scorer with identity rotations reproduces the uncompensated
# baseline's summary figure exactly, but the per-window
# distances differ by up to ~4e-4 px. That is NOT a plumbing bug, and the two
# tests below pin down why:
#
#   * the CMC transform under identity rotations moves nothing (to 1e-9 px;
#     see the identity test for why that is a tolerance and not bit-identity),
#     and
#   * the raw ground truth read from annotations is bit-identical to the future
#     stored in PPT's own pickle,
#
# so no coordinate is being altered. The residual is float32 association order:
# the baseline computes ||out - (fut - init)|| in float32 on the GPU, while the
# CMC path computes ||(out + init) - fut||. Algebraically identical, rounded
# differently. Hence the eval-level tolerance is float32 epsilon, not zero.
# --------------------------------------------------------------------------- #
PICKLE = ROOT / "PPT" / "data" / "social_sdd_test_4096_0_100.pickle"
META = ROOT / "PPT" / "data" / "social_sdd_test_4096_0_100_meta.npz"


def test_decompensate_under_identity_leaves_predictions_in_place(cfg):
    """Round-off only, not a coordinate change. See the note on the identity
    test above for why this is a tolerance rather than bit-identity."""
    save_rotations("bit01", synthetic(WIN + 10, mode="identity"), cfg)
    probe = np.random.default_rng(0).uniform(100.0, 3000.0, size=(6, T_PRED, 2))
    back = pipeline.decompensate_modes(probe, "bit01", 1, T_OBS, cfg)
    assert np.abs(back - probe).max() < 1e-9


@pytest.mark.skipif(not META.exists(), reason="run prediction.to_ppt_format first")
def test_raw_gt_matches_the_pickle_exactly():
    """The eval GT source must agree with what PPT itself was scored against.

    If these ever diverge, CMC results become incomparable with the baseline for a
    reason that has nothing to do with camera motion.
    """
    import pickle as _pickle
    side = np.load(META)
    if bool(side.get("cmc_enabled", False)):
        pytest.skip("pickle is CMC-compensated; its future is not raw GT by design")
    meta = side["meta"]
    batches, _ = _pickle.load(open(PICKLE, "rb"))
    windows = batches[0][:, :, 2:4].astype(np.float64)
    for i, (seq, tid, start) in enumerate(meta):
        gt = pipeline.raw_future_gt(int(seq), int(tid), int(start), T_OBS, T_PRED)
        assert np.array_equal(gt, windows[i, T_OBS:]), f"window {i} disagrees"


# --------------------------------------------------------------------------- #
# Fidelity to the reference formulation
# --------------------------------------------------------------------------- #
def _guide_compensate_trajectory(trajectory, rotations, K):
    """Verbatim transcription of the per-point reference loop in docs/cmc.md.

    Kept exactly as written -- per-point Python loop and all -- so the optimised
    implementation can be checked against it rather than against my own
    restatement of it.
    """
    T = trajectory.shape[0]
    R_ref = rotations[0]
    K_inv = np.linalg.inv(K)
    compensated = np.zeros_like(trajectory)
    for t in range(T):
        H = K @ R_ref @ np.linalg.inv(rotations[t]) @ K_inv
        p = np.array([trajectory[t, 0], trajectory[t, 1], 1.0])
        p_comp = H @ p
        compensated[t] = p_comp[:2] / p_comp[2]
    return compensated


def test_matches_the_guide_reference_implementation(traj):
    """Our vectorised compensate_trajectory must equal the reference loop.

    The loop is the specification. Ours is an einsum because it runs
    per mode per window, and speed is no excuse for drifting from the spec.
    """
    for mode in ("sinusoidal", "pan_jitter", "identity"):
        R = synthetic(WIN, mode=mode)
        ours = compensate_trajectory(traj, R, K_4K)
        theirs = _guide_compensate_trajectory(traj.copy(), R, K_4K)
        assert np.abs(ours - theirs).max() < 1e-9, f"diverges from the reference on {mode}"


def test_guide_reference_frame_convention_is_the_first_frame(traj):
    """§5.2: reference frame = rotations[0]. The first point must not move."""
    R = synthetic(WIN, mode="sinusoidal")
    out = compensate_trajectory(traj, R, K_4K)
    assert np.allclose(out[0], traj[0], atol=1e-9)


# --------------------------------------------------------------------------- #
# reference="last" and input-only CMC
# --------------------------------------------------------------------------- #
def test_round_trip_with_last_reference(traj, cfg):
    """The 'last' reference convention must also round-trip exactly."""
    from dataclasses import replace as _replace
    last_cfg = _replace(cfg, reference="last")
    save_rotations("rt_last", synthetic(WIN + 10), last_cfg)
    comp = pipeline.compensate_window(traj, "rt_last", 1, last_cfg, t_obs=T_OBS)
    assert np.abs(comp - traj).max() > 1.0, "test is vacuous if nothing moved"
    assert np.allclose(comp[T_OBS - 1], traj[T_OBS - 1], atol=1e-9), \
        "the last observed frame is the reference and must not move"
    R = pipeline.rotations_for("rt_last", last_cfg)
    R_win = R[:WIN]
    R_ref = R_win[T_OBS - 1]
    back = inverse_compensate(comp, R_win, K_for("rt_last", last_cfg), R_ref=R_ref)
    assert np.abs(back - traj).max() < 1e-6


def test_compensate_observation_only_leaves_future_raw(traj, cfg):
    """Input-only CMC must leave prediction target frames untouched."""
    save_rotations("obs01", synthetic(WIN + 10), cfg)
    out = pipeline.compensate_observation_only(traj, "obs01", 1, T_OBS, cfg)
    assert np.array_equal(out[T_OBS:], traj[T_OBS:]), \
        "future frames must be bit-identical to raw input"
    assert not np.array_equal(out[:T_OBS], traj[:T_OBS]), \
        "observation should differ (camera is moving)"
    assert np.allclose(out[T_OBS - 1], traj[T_OBS - 1], atol=1e-9), \
        "last observed frame is the reference and must not move"


def test_config_validates_fps_and_extrapolation():
    """Bad fps or unknown extrapolation mode must be rejected early."""
    with pytest.raises(ValueError, match="fps must be positive"):
        CMCConfig(fps=0).validate()
    with pytest.raises(ValueError, match="fps must be positive"):
        CMCConfig(fps=-1).validate()
    with pytest.raises(ValueError, match="unknown extrapolation"):
        CMCConfig(extrapolation="nonexistent").validate()


def test_config_load_rejects_non_dict(tmp_path):
    """A JSON array or scalar is not a valid config."""
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        CMCConfig.load(str(bad))


# --------------------------------------------------------------------------- #
# Per-sequence intrinsics
#
# K used to be one matrix for the whole run. It cannot be: the dataset mixes
# three cameras at two resolutions, so focal length varies by a factor of two
# and the principal point by the same. These tests pin the lookup down.
# --------------------------------------------------------------------------- #
def test_k_is_per_sequence_not_per_run(tmp_path):
    """Two sequences with different cameras must get different matrices."""
    path = write_intrinsics(tmp_path / "ci.json", {
        "train/0001": (K_4K, (3840, 2160)),
        "train/0002": (K_FHD, (1920, 1080)),
    })
    cfg = CMCConfig(enabled=True, intrinsics=path)
    assert np.array_equal(K_for("0001", cfg), K_4K)
    assert np.array_equal(K_for("0002", cfg), K_FHD)


def test_split_disambiguates_repeated_sequence_ids(tmp_path):
    """train/0001 and pub_test/0001 are different footage from different rigs.

    A bare sequence id cannot identify a camera, so cfg.split has to select.
    Getting this wrong would silently de-rotate with the wrong focal length.
    """
    path = write_intrinsics(tmp_path / "ci.json", {
        "train/0001": (K_4K, (3840, 2160)),
        "pub_test/0001": (K_FHD, (1920, 1080)),
    })
    train = CMCConfig(enabled=True, intrinsics=path, split="train")
    test = CMCConfig(enabled=True, intrinsics=path, split="pub_test")
    assert np.array_equal(K_for("0001", train), K_4K)
    assert np.array_equal(K_for("0001", test), K_FHD)


def test_unknown_sequence_error_mentions_the_split(tmp_path):
    path = write_intrinsics(tmp_path / "ci.json", {"train/0001": (K_4K, (3840, 2160))})
    cfg = CMCConfig(enabled=True, intrinsics=path, split="pub_test")
    with pytest.raises(KeyError, match="no camera intrinsics"):
        K_for("0001", cfg)


def test_principal_point_must_match_the_frame(tmp_path):
    """The check that would have caught the old placeholder K.

    Its centre was (1920, 1080) -- correct for 4K, but off by a factor of two
    on the sequences shot at 1920x1080.
    """
    path = write_intrinsics(tmp_path / "ci.json", {
        "train/0001": (K_4K, (1920, 1080)),          # 4K centre, 1080p frame
    })
    cfg = CMCConfig(enabled=True, intrinsics=path)
    with pytest.raises(ValueError, match="principal point"):
        K_for("0001", cfg)


def test_malformed_entries_name_the_sequence(tmp_path):
    """A bad intrinsics file must say which sequence is wrong, not just fail."""
    cases = {
        "no-K": ({"resolution": [3840, 2160]}, "has no 'K'"),
        "text-K": ({"K": "nope"}, "not numeric"),
        "small-K": ({"K": [[1, 0], [0, 1]]}, "must be 3x3"),
        "nan-K": ({"K": [[1, 0, 0], [0, None, 0], [0, 0, 1]]}, "non-finite"),
    }
    for name, (entry, message) in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"sequences": {"train/0001": entry}}))
        cfg = CMCConfig(enabled=True, intrinsics=path)
        with pytest.raises(ValueError, match=message) as excinfo:
            K_for("0001", cfg)
        assert "train/0001" in str(excinfo.value), f"{name} does not name the sequence"


def test_missing_intrinsics_file_is_actionable(tmp_path):
    cfg = CMCConfig(enabled=True, intrinsics=tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="SMOT4SB_ROOT"):
        K_for("0001", cfg)


def test_returned_k_is_read_only(tmp_path):
    """It is cached and shared, so a caller must not be able to corrupt it."""
    path = write_intrinsics(tmp_path / "ci.json", {"train/0001": (K_4K, (3840, 2160))})
    K = K_for("0001", CMCConfig(enabled=True, intrinsics=path))
    with pytest.raises(ValueError):
        K[0, 0] = 1.0


@pytest.mark.parametrize("split", ["", "train/0001"])
def test_config_rejects_malformed_split(split):
    # Split names are not enumerated: an unknown one fails at K lookup with
    # the split in the message (test_unknown_sequence_error_mentions_the_split).
    with pytest.raises(ValueError, match="bare split name"):
        CMCConfig(split=split).validate()


REAL_INTRINSICS = paths.SMOT4SB / "camera_intrinsics.json"


@pytest.mark.skipif(not REAL_INTRINSICS.is_file(), reason="dataset not mounted")
def test_every_training_sequence_resolves_against_the_real_dataset():
    """No training sequence may be missing intrinsics.

    A dataset drop that adds a sequence without a camera entry would otherwise
    surface as a KeyError in the middle of a training run.
    """
    from prediction.build_dataset import load_tracks
    cfg = CMCConfig(enabled=True, intrinsics=REAL_INTRINSICS)
    sequences = sorted({seq for seq, _ in load_tracks()})
    assert sequences, "no tracks loaded"
    focals = {seq: float(K_for(seq, cfg)[0, 0]) for seq in sequences}
    # If this ever collapses to one value, the per-sequence lookup has silently
    # stopped varying and a run-wide K would look correct again.
    assert len(set(focals.values())) > 1, "expected several cameras in train"
    assert 3840.0 not in set(focals.values()), \
        "3840 was the removed placeholder focal; no real camera has it"


# --------------------------------------------------------------------------- #
# Input-only CMC (mode="input_only")
#
# The observation is compensated into the LAST observed frame; the target is
# left in raw pixels. The tests below pin down the property that makes scoring
# work: at the reference the homography is exactly the identity, so the anchor
# PPT predicts from keeps its true pixel value and no decompensation is needed.
# --------------------------------------------------------------------------- #
@pytest.fixture
def io_cfg(cfg):
    """The same config in input-only mode. with_mode moves the reference."""
    return cfg.with_mode("input_only").validate()


def test_input_only_anchor_keeps_its_raw_pixel_value(traj, io_cfg):
    """The last observed point must not move. This is what lets the scorer skip
    decompensation: PPT predicts relative to that anchor, so if the anchor is
    raw then so is `output + anchor`."""
    save_rotations("obs01", synthetic(WIN + 10), io_cfg)
    out = pipeline.compensate_observation_only(traj, "obs01", 1, T_OBS, io_cfg)
    assert np.allclose(out[T_OBS - 1], traj[T_OBS - 1], atol=1e-9)
    assert np.array_equal(out[T_OBS:], traj[T_OBS:]), "target must stay raw"
    assert np.abs(out[:T_OBS - 1] - traj[:T_OBS - 1]).max() > 1.0, \
        "test is vacuous if the observation did not move"


def test_input_only_needs_no_future_rotation(io_cfg):
    """It must work when rotations stop at the last observed frame.

    Full CMC has to extrapolate 150 future rotations; input-only must not touch
    them at all, and the cheapest proof is that it succeeds with none available.
    """
    save_rotations("obsshort", synthetic(T_OBS), io_cfg)
    w = np.random.default_rng(0).uniform(200.0, 1800.0, size=(WIN, 2))
    out = pipeline.compensate_observation_only(w, "obsshort", 1, T_OBS, io_cfg)
    assert np.array_equal(out[T_OBS:], w[T_OBS:])
    with pytest.raises(ValueError, match="obsshort"):
        pipeline.compensate_window(w, "obsshort", 1, replace(io_cfg, mode="full"))


def test_input_only_disabled_is_still_a_noop(traj, tmp_path):
    off = CMCConfig(enabled=False, mode="input_only", reference="last",
                    rotation_dir=tmp_path / "nope")
    assert pipeline.compensate_observation_only(traj, "x", 1, T_OBS, off) is traj


def test_with_mode_moves_the_reference(cfg):
    """One helper owns the rule, so the two CLIs cannot drift apart on it."""
    assert cfg.mode == "full" and cfg.reference == "first"
    io = cfg.with_mode("input_only")
    assert (io.mode, io.reference) == ("input_only", "last")
    # Going back leaves the reference where input_only put it rather than
    # guessing a previous value, and full accepts either reference.
    assert io.with_mode("full").mode == "full"
    io.with_mode("full").validate()


def test_input_only_requires_the_last_reference():
    """reference='first' would move the anchor while the target stayed raw,
    leaving a per-window offset. Silently ignoring the field would be worse."""
    with pytest.raises(ValueError, match="requires reference='last'"):
        CMCConfig(mode="input_only", reference="first").validate()
    with pytest.raises(ValueError, match="mode must be"):
        CMCConfig(mode="observation").validate()


def test_full_mode_is_unchanged_by_the_switch(traj, cfg):
    """The default is still full CMC, behaving exactly as before."""
    assert cfg.mode == "full"
    save_rotations("stillfull", synthetic(WIN + 10), cfg)
    comp = pipeline.compensate_window(traj, "stillfull", 1, cfg)
    assert np.allclose(comp[0], traj[0], atol=1e-9), "first frame is the reference"
    assert np.abs(comp[T_OBS:] - traj[T_OBS:]).max() > 1.0, \
        "full CMC must transform the target too"


def test_sidecar_round_trips_the_mode(tmp_path):
    """score_ppt reads the mode back out of the npz as a bare string.

    The guard against scoring a pickle in the wrong coordinate space depends on
    this, and numpy stores a str as a 0-d unicode array, so the read side has to
    go back through str(). Pinning it here because a silent change would turn
    the guard into a comparison that never matches.
    """
    path = tmp_path / "meta.npz"
    for mode in ("full", "input_only", "none"):
        np.savez_compressed(path, meta=np.zeros((2, 3), dtype=np.int64),
                            cmc_enabled=True, cmc_mode=mode)
        loaded = np.load(path)
        assert str(loaded["cmc_mode"]) == mode
        assert bool(loaded.get("cmc_enabled", False)) is True

    # A sidecar written before the mode existed must read as full CMC, not crash.
    np.savez_compressed(path, meta=np.zeros((2, 3), dtype=np.int64), cmc_enabled=True)
    loaded = np.load(path)
    assert "cmc_mode" not in loaded


# --------------------------------------------------------------------------- #
# A rotation file must belong to the footage it is named for
# --------------------------------------------------------------------------- #
def test_check_covers_requires_one_rotation_per_frame():
    """Too short fails later with a confusing frame-range error; too long is read
    without complaint. Both mean the file describes different footage."""
    from cmc.rotations import check_covers
    R = synthetic(100, mode="identity")
    assert check_covers(R, "v", 100) is R
    with pytest.raises(ValueError, match="has 100 frames but the sequence has 120"):
        check_covers(R, "v", 120)
    with pytest.raises(ValueError, match="has 100 frames but the sequence has 90"):
        check_covers(R, "v", 90)


@pytest.mark.skipif(not (paths.SMOT4SB / "train").is_dir(), reason="dataset not mounted")
def test_stale_rotation_file_is_refused_for_a_real_sequence(tmp_path):
    """A cached rotation file sized for other footage must be refused, not
    silently read: a file longer than the sequence raises no other error."""
    from cmc.rotations import frame_count
    seq = sorted(p.name for p in (paths.SMOT4SB / "train").iterdir() if p.is_dir())[0]
    n = frame_count(seq, "train")
    cfg = CMCConfig(enabled=True, source="rotations", rotation_dir=tmp_path, split="train",
                    intrinsics=write_intrinsics(tmp_path / "ci.json",
                                                {f"train/{seq}": (K_4K, (3840, 2160))}))
    save_rotations(seq, synthetic(n + 5, mode="identity"), cfg)   # the old convention
    with pytest.raises(ValueError, match=f"video {seq}: rotation file has {n + 5}"):
        pipeline.rotations_for(seq, cfg)
    pipeline._rotations_cached.cache_clear()
    save_rotations(seq, synthetic(n, mode="identity"), cfg)
    assert len(pipeline.rotations_for(seq, cfg)) == n


def test_synthetic_ids_skip_the_ownership_check(cfg):
    """Ids with no image directory are the tests' own inventions; there is
    nothing to check them against, and they must keep working."""
    from cmc.rotations import frame_count
    assert frame_count("id01", "train") is None
    save_rotations("id01", synthetic(7, mode="identity"), cfg)
    assert len(pipeline.rotations_for("id01", cfg)) == 7


# --------------------------------------------------------------------------- #
# Rotations from the dataset's imu.csv
# --------------------------------------------------------------------------- #
def _write_imu_csv(path, R, retracted=0):
    import csv as _csv
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["frame", "video_frame", "log_time_s", "yaw_deg", "vn_mps", "ve_mps",
                    "vd_mps", "gimbal_retracted"] + [f"R{i}{j}" for i in range(3) for j in range(3)])
        for t, Rt in enumerate(R, start=1):
            w.writerow([t, t + 100, t / 30.0, 0.0, 0.0, 0.0, 0.0, retracted]
                       + [f"{v:.6f}" for v in Rt.reshape(-1)])       # six decimals, like the dataset


def test_imu_csv_loads_and_survives_six_decimal_rounding(tmp_path):
    """The dataset stores the body attitude to 6 decimals, which fails
    check_rotations as-is; the loader must project back onto the rotation group.
    With the gimbal retracted the camera is fixed to the body, so the result is
    the body rotation mapped to camera axes, to well inside that rounding.
    (Physical correctness is tested in test_cmc_physics.py.)"""
    from cmc.rotations import CAMERA_IN_BODY, load_imu_rotations
    R = synthetic(50, mode="sinusoidal")
    _write_imu_csv(tmp_path / "imu.csv", R, retracted=1)
    out = load_imu_rotations(tmp_path / "imu.csv")
    check_rotations(out)                                    # orthonormal, det +1
    assert out.shape == (50, 3, 3)
    expected = np.transpose(R @ CAMERA_IN_BODY, (0, 2, 1))
    assert np.abs(out - expected).max() < 5e-6


def test_imu_csv_rejects_gaps_in_the_frame_column(tmp_path):
    from cmc.rotations import load_imu_rotations
    R = synthetic(10, mode="identity")
    _write_imu_csv(tmp_path / "imu.csv", R)
    lines = (tmp_path / "imu.csv").read_text().splitlines()
    del lines[5]                                            # drop frame 5
    (tmp_path / "imu.csv").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="frame column"):
        load_imu_rotations(tmp_path / "imu.csv")


@pytest.mark.skipif(not (paths.SMOT4SB / "train").is_dir(), reason="dataset not mounted")
def test_every_imu_sequence_loads_and_covers_its_frames():
    """Every IMU log the intrinsics file promises, in the splits present here,
    must load through the real path -- no cached .npz, so load_rotations falls
    through to imu.csv -- and pass the ownership check."""
    from cmc.rotations import imu_path
    S = json.loads((paths.SMOT4SB / "camera_intrinsics.json").read_text())["sequences"]
    imu = [k.split("/") for k, v in S.items()
           if v["has_imu"] and (paths.SMOT4SB / k.split("/")[0]).is_dir()]
    assert imu, "intrinsics file lists no IMU sequence in the splits present"
    for split, seq in imu:
        assert imu_path(seq, split) is not None
        cfg = CMCConfig(enabled=True, split=split, rotation_dir=paths.DATA / "no-such-dir")
        R = pipeline.rotations_for(seq, cfg)
        assert R.shape[1:] == (3, 3) and len(R) > 0
    pipeline._rotations_cached.cache_clear()


def test_no_source_names_both_options(tmp_path):
    """A sequence with neither a cached file nor an IMU log must say so."""
    from cmc.rotations import load_rotations
    with pytest.raises(SystemExit, match="no rotation source"):
        load_rotations("nonexistent", CMCConfig(rotation_dir=tmp_path))


# --------------------------------------------------------------------------- #
# The LSTM baseline's window loader under input-only CMC
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (paths.SMOT4SB / "train").is_dir() or not paths.SPLIT.is_file(),
                    reason="dataset or split.json not available")
def test_lstm_loader_restricts_to_imu_sequences_for_both_arms():
    """--only-with-rotations must give the raw arm and the compensated arm the
    same windows, or 'CMC vs none' also compares the IMU sequences to all."""
    from prediction.learned_models import load_split
    raw = load_split(True, str(paths.SPLIT), 210, None, only_with_rotations=True)
    if len(raw[0]) == 0:
        pytest.skip("no validation sequence has a rebuilt imu.csv "
                    "(set SMOTP4SB_IMU_DIR or install the rebuilt files)")
    cfg = CMCConfig(enabled=True).with_mode("input_only").validate()
    cmc = load_split(True, str(paths.SPLIT), 210, cfg, only_with_rotations=True)
    assert len(raw[0]) == len(cmc[0]) > 0
    # input-only: the target is raw in both arms, so it must be identical
    assert np.array_equal(raw[1], cmc[1])
    # ...and the origin (last observed point) is the raw anchor in both
    assert np.allclose(raw[2], cmc[2], atol=1e-4)
    # ...while the observation differs wherever the camera actually moved
    assert not np.array_equal(raw[0], cmc[0])
