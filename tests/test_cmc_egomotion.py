"""IMU-driven CMC (cmc/egomotion.py and its wiring in cmc/pipeline.py)."""

import csv
from dataclasses import replace

import numpy as np
import pytest

from cmc import pipeline
from cmc.compensate import apply_homographies
from cmc.config import CMCConfig
from cmc.egomotion import (EgoMotionModel, LAGS, feature_names, features, homographies_to_ref,
                           imu_window, load_imu, similarity)

COLUMNS = ["frame", "yaw_deg", "pitch_deg", "gimbal_retracted", "gimbal_yaw_cmd", "gimbal_pitch_cmd", "zoom_cmd"]
W, H = 1920, 1080
C = (960.0, 540.0)


def imu_dict(n: int, rng: np.random.Generator) -> dict:
    return {"frame": np.arange(1, n + 1, dtype=float),
            "yaw_deg": np.cumsum(rng.normal(0, 0.2, n)) % 360,
            "pitch_deg": rng.normal(-2, 0.3, n),
            "gimbal_retracted": np.zeros(n),
            "gimbal_yaw_cmd": np.round(rng.uniform(-1, 1, n) * (rng.random(n) < 0.2), 3),
            "gimbal_pitch_cmd": np.round(rng.uniform(-1, 1, n) * (rng.random(n) < 0.2), 3),
            "zoom_cmd": rng.choice([-1.0, 0.0, 1.0], n, p=[0.05, 0.9, 0.05])}


def write_imu(path, imu: dict) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for i in range(len(imu["frame"])):
            w.writerow([int(imu["frame"][i])] + [imu[c][i] for c in COLUMNS[1:]])


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_features_use_only_the_rows_they_are_given():
    """Track 2 passes the observed rows only; nothing outside them may leak in."""
    rng = np.random.default_rng(0)
    imu = imu_dict(200, rng)
    win = imu_window(imu, 51, 60)
    other = {k: v.copy() for k, v in imu.items()}
    for k in ("gimbal_yaw_cmd", "gimbal_pitch_cmd", "zoom_cmd", "yaw_deg"):
        other[k][:50] = rng.normal(size=50)
        other[k][110:] = rng.normal(size=90)
    assert np.array_equal(features(win), features(imu_window(other, 51, 60)))


def test_feature_lags_point_at_the_right_rows():
    imu = imu_dict(30, np.random.default_rng(1))
    imu["gimbal_yaw_cmd"][:] = 0.0
    imu["gimbal_yaw_cmd"][10] = 1.0
    X = features(imu)
    names = feature_names()
    for lag in (-8, 0, 8):
        col = names.index(f"gy[{lag:+d}]")
        assert np.flatnonzero(X[:, col]).tolist() == [10 - lag]      # x[t] = v[t + lag]
    assert X.shape == (29, sum(hi - lo + 1 for lo, hi in LAGS.values()))


def test_body_yaw_step_unwraps_across_north():
    imu = imu_dict(4, np.random.default_rng(2))
    imu["yaw_deg"] = np.array([359.0, 1.0, 3.0, 5.0])
    X = features(imu)
    col = feature_names().index("body[+0]")
    assert np.allclose(X[:, col], -np.radians(2.0))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_fit_recovers_a_linear_mapping_and_round_trips_through_json(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(4000, len(feature_names())))
    Wt = np.zeros((X.shape[1], 3))
    Wt[feature_names().index("gy[+2]"), 0] = -12.0
    Wt[feature_names().index("gp[+1]"), 1] = 9.0
    Wt[feature_names().index("zc[+0]"), 2] = 0.02
    Y = X @ Wt + 0.1
    m = EgoMotionModel.fit(X, Y, lam=1e-6)
    assert np.allclose(m.predict(X), Y, atol=1e-4)
    damped = EgoMotionModel.fit(X, Y, lam=1e-6, gain=0.5)
    assert np.allclose(damped.predict(X), 0.5 * Y, atol=1e-4)
    m.save(tmp_path / "m.json")
    m2 = EgoMotionModel.load(tmp_path / "m.json")
    assert np.allclose(m2.predict(X[:10]), m.predict(X[:10]))
    # pixel outputs scale with the frame width, ln s does not
    half = m.predict(X[:10], width=960)
    assert np.allclose(half[:, :2], m.predict(X[:10])[:, :2] / 2) and np.allclose(half[:, 2], m.predict(X[:10])[:, 2])


def test_zero_weight_transitions_do_not_bias_the_fit():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2000, len(feature_names())))
    Y = np.zeros((2000, 3))
    Y[:100] = 500.0                                      # failed measurements
    w = np.ones(2000); w[:100] = 0.0
    m = EgoMotionModel.fit(X, Y, w)
    assert np.abs(m.predict(X[100:])).max() < 1e-6


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def test_homographies_are_identity_at_the_reference_and_invert_cleanly():
    rng = np.random.default_rng(5)
    motion = np.stack([rng.normal(0, 5, 59), rng.normal(0, 5, 59), rng.normal(0, 0.01, 59)], 1)
    Hs = homographies_to_ref(motion, 59, C)
    assert np.allclose(Hs[59], np.eye(3))
    H0 = homographies_to_ref(motion, 0, C)
    # frame t -> 59 -> 0 equals frame t -> 0
    assert np.allclose(np.einsum("ij,tjk->tik", H0[59], Hs), H0, atol=1e-9)


def test_a_point_carried_by_the_camera_is_held_still():
    """Synthesise a world point seen through the per-frame similarities and
    check compensation puts every observation where the reference frame saw it."""
    rng = np.random.default_rng(6)
    motion = np.stack([rng.normal(0, 8, 59), rng.normal(0, 4, 59), rng.normal(0, 0.02, 59)], 1)
    p = np.array([700.0, 300.0]); seen = [p]
    for dx, dy, ls in motion:
        q = similarity(dx, dy, ls, *C) @ np.array([*seen[-1], 1.0])
        seen.append(q[:2])
    seen = np.array(seen)
    comp = apply_homographies(seen, homographies_to_ref(motion, 59, C))
    assert np.ptp(seen, axis=0).max() > 50
    assert np.abs(comp - seen[-1]).max() < 1e-6


# --------------------------------------------------------------------------- #
# imu.csv and the pipeline
# --------------------------------------------------------------------------- #
def test_old_format_imu_is_refused(tmp_path):
    p = tmp_path / "imu.csv"
    p.write_text("frame,video_frame,log_time_s,yaw_deg,R00\n1,1,0.0,10,1\n")
    with pytest.raises(ValueError, match="old-format"):
        load_imu(p)


def test_frame_column_must_be_contiguous(tmp_path):
    imu = imu_dict(5, np.random.default_rng(7))
    imu["frame"] = np.array([1, 2, 4, 5, 6], float)
    write_imu(tmp_path / "imu.csv", imu)
    with pytest.raises(ValueError, match="frame column"):
        load_imu(tmp_path / "imu.csv")


@pytest.fixture
def ego_cfg(tmp_path):
    """A one-sequence dataset (imu.csv + intrinsics) and a model that maps the
    gimbal yaw command straight to a horizontal pan."""
    rng = np.random.default_rng(8)
    imu = imu_dict(240, rng)
    d = tmp_path / "imu" / "train" / "0001"; d.mkdir(parents=True)
    write_imu(d / "imu.csv", imu)
    intr = tmp_path / "intr.json"
    intr.write_text('{"sequences": {"train/0001": {"camera": "x", "resolution": [1920, 1080], '
                    '"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]}}}')
    F = len(feature_names())
    Wm = np.zeros((F, 3)); Wm[feature_names().index("gy[+0]"), 0] = -20.0
    model = EgoMotionModel(mean=np.zeros(F), scale=np.ones(F), W=Wm, bias=np.zeros(3), gain=1.0)
    model.save(tmp_path / "ego.json")
    pipeline._imu_cached.cache_clear(); pipeline._model_cached.cache_clear()
    cfg = CMCConfig(enabled=True, imu_dir=tmp_path / "imu", intrinsics=intr,
                    egomotion_model=tmp_path / "ego.json").with_mode("input_only").validate()
    return cfg, imu


def test_input_only_compensation_matches_the_model_and_keeps_the_anchor(ego_cfg):
    cfg, imu = ego_cfg
    rng = np.random.default_rng(9)
    win = np.cumsum(rng.normal(0, 3, (210, 2)), 0) + [800, 500]
    out = pipeline.compensate_observation_only(win, "0001", 31, 60, cfg)
    assert np.array_equal(out[60:], win[60:])                        # target untouched
    assert np.allclose(out[59], win[59])                             # anchor is the identity
    gy = imu["gimbal_yaw_cmd"][30:89]
    # undoing the predicted pans: frame t is shifted by the pans still to come
    expected_dx = np.array([(-20.0 * gy[t:]).sum() for t in range(59)])
    assert np.allclose(out[:59, 0] - win[:59, 0], expected_dx)
    assert np.allclose(out[:59, 1], win[:59, 1])


def test_full_mode_round_trip_holds_the_camera_over_the_horizon(ego_cfg):
    cfg, _ = ego_cfg
    full = replace(cfg, mode="full", reference="first").validate()
    modes = np.random.default_rng(10).normal(900, 50, (6, 150, 2))
    raw = pipeline.decompensate_modes(modes, "0001", 31, 60, full)
    # no future IMU is read: every future frame gets the last observed frame's map
    shift = raw - modes
    assert np.allclose(shift, shift[:, :1])


def test_disabled_cmc_is_a_no_op(ego_cfg):
    cfg, _ = ego_cfg
    off = replace(cfg, enabled=False)
    win = np.random.default_rng(11).normal(size=(210, 2))
    assert pipeline.compensate_observation_only(win, "0001", 1, 60, off) is win


def test_motion_source_needs_a_rebuilt_imu(ego_cfg, tmp_path):
    cfg, _ = ego_cfg
    assert pipeline.has_motion_source("0001", cfg)
    assert not pipeline.has_motion_source("0002", cfg)
    old = tmp_path / "imu" / "train" / "0003"; old.mkdir(parents=True)
    (old / "imu.csv").write_text("frame,video_frame,log_time_s,yaw_deg,R00\n1,1,0,0,1\n")
    assert not pipeline.has_motion_source("0003", cfg)
