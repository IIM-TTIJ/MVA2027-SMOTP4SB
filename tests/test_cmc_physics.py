"""CMC against physics, not against itself.

The earlier tests checked that compensation round-trips and is the identity
under identity rotations. They could not catch an axis error: imu.csv holds
the drone body's attitude (body -> NED, ZYX Euler), and reading it directly as
a world-to-camera rotation turned yaw into image roll and pitch/roll into
pans. These tests project a world-fixed point with a pinhole model written
from first principles, independent of cmc/, and require the compensation to
put it back where the reference frame saw it.

Camera model: x right, y down, z along the optical axis. The gimbal follows
the drone's yaw and holds pitch and roll level, so normally the camera points
along the heading, horizontal; when the gimbal is retracted the camera is
fixed to the airframe, looking along the body's forward axis.
"""

import csv
from pathlib import Path

import numpy as np

from cmc.compensate import compensate_trajectory
from cmc.rotations import load_imu_rotations

F, CX, CY = 2248.0, 1920.0, 1080.0
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])


def body_to_ned(roll, pitch, yaw):
    """As SMOT4SB/tools/extract_imu.py writes it: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return Rz @ Ry @ Rx


def project(X, roll, pitch, yaw, retracted):
    """Pixel of NED direction X, from first principles (no cmc/ code)."""
    if retracted:
        Rb = body_to_ned(roll, pitch, yaw)
        fwd, right, down = Rb[:, 0], Rb[:, 1], Rb[:, 2]
    else:
        y = np.radians(yaw)
        fwd = np.array([np.cos(y), np.sin(y), 0.0])
        right = np.array([-np.sin(y), np.cos(y), 0.0])
        down = np.array([0.0, 0.0, 1.0])
    z = X @ fwd
    return np.array([CX + F * (X @ right) / z, CY + F * (X @ down) / z])


def write_imu(path: Path, attitudes, retracted=0):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "video_frame", "log_time_s", "yaw_deg", "vn_mps", "ve_mps",
                    "vd_mps", "gimbal_retracted"] + [f"R{i}{j}" for i in range(3) for j in range(3)])
        for t, (r, p, y) in enumerate(attitudes, start=1):
            R = body_to_ned(r, p, y)
            w.writerow([t, t, t / 30.0, f"{y % 360:.4f}", 0, 0, 0, retracted]
                       + [f"{v:.6f}" for v in R.reshape(-1)])


def compensated(tmp_path, attitudes, X, retracted=0):
    write_imu(tmp_path / "imu.csv", attitudes, retracted)
    R = load_imu_rotations(tmp_path / "imu.csv")
    seen = np.array([project(X, *a, retracted) for a in attitudes])
    return seen, compensate_trajectory(seen, R, K)


def test_the_first_principles_model_pans_left_when_the_drone_yaws_right():
    """Sanity of the reference model itself: yawing right by 1 deg moves a
    static point at the image centre left by f*tan(1 deg) = 39.2 px."""
    X = np.array([1.0, 0.0, 0.0])                       # straight ahead, north
    a = project(X, 0, 0, 0, retracted=0)
    b = project(X, 0, 0, 1, retracted=0)
    assert np.allclose(a, [CX, CY])
    assert np.isclose(b[0] - a[0], -F * np.tan(np.radians(1)), atol=1e-6)
    assert np.isclose(b[1], a[1])


def test_yaw_is_compensated_as_a_horizontal_pan(tmp_path):
    X = np.array([1.0, 0.15, -0.05])
    att = [(0, 0, 10 + 0.5 * t) for t in range(20)]     # steady right turn
    seen, comp = compensated(tmp_path, att, X)
    assert np.ptp(seen[:, 0]) > 300                     # the raw point moves a lot
    assert np.abs(comp - seen[0]).max() < 0.05          # compensation holds it still


def test_body_pitch_and_roll_do_not_move_the_image_while_the_gimbal_holds(tmp_path):
    X = np.array([1.0, -0.2, 0.1])
    att = [(3 * np.sin(t / 3), 4 * np.cos(t / 4), 30.0) for t in range(20)]
    seen, comp = compensated(tmp_path, att, X)
    assert np.ptp(seen, axis=0).max() < 1e-9            # physics: nothing moves
    assert np.abs(comp - seen[0]).max() < 0.05          # and nothing is injected


def test_with_the_gimbal_retracted_the_full_attitude_applies(tmp_path):
    """Nose up by 1 deg: the camera looks up, a static point moves down."""
    X = np.array([1.0, 0.0, 0.0])
    a = project(X, 0, 0, 0, retracted=1)
    b = project(X, 0, 1, 0, retracted=1)
    assert np.isclose(b[1] - a[1], F * np.tan(np.radians(1)), atol=1e-6)
    att = [(1.5 * np.sin(t / 2), 0.6 * t, 20 + 0.3 * t) for t in range(20)]
    seen, comp = compensated(tmp_path, att, np.array([1.0, 0.1, 0.05]), retracted=1)
    assert np.ptp(seen, axis=0).max() > 100
    assert np.abs(comp - seen[0]).max() < 0.05
