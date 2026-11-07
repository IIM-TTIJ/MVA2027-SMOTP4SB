"""Lay out a test split the way submitted code receives it in the private test.

    python tools/private_test/package_inputs.py --track 1 --split pub_test \
        --data $SMOTP4SB_DATA --out /tmp/pub_test_t1_input
    python tools/private_test/package_inputs.py --track 2 --split pub_test \
        --data $SMOTP4SB_DATA --out /tmp/pub_test_t2_input

Track 1 gets whole sequences. Track 2 gets one directory per window holding only
its 60 observed frames (original names) and their imu.csv rows, so that nothing
after the observation is in reach (docs/private_test.md). Windows start at frame 1
and every 150 frames while a whole 60 + 150 span fits, as the scorer cuts them;
all of them are given, whether or not a bird is visible.

Frames are hard-linked where the filesystem allows and copied otherwise.
camera_intrinsics.json is reduced to the split's sequences, same format.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

T_OBS, T_PRED = 60, 150
FRAME_EXTS = (".jpg", ".jpeg", ".png")


def place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def frames_of(seq_dir: Path) -> list[Path]:
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)


def imu_rows(path: Path, first: int, last: int, out: Path) -> int:
    with open(path, newline="") as fin, open(out, "w", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        header = next(r)
        w.writerow(header)
        col = header.index("frame")
        n = 0
        for row in r:
            if first <= int(row[col]) <= last:
                w.writerow(row)
                n += 1
    return n


def windows(n_frames: int) -> list[int]:
    return list(range(1, n_frames - T_OBS - T_PRED + 2, T_PRED))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", type=int, choices=(1, 2), required=True)
    ap.add_argument("--split", required=True, help="split directory under --data, e.g. pub_test")
    ap.add_argument("--data", required=True, help="dataset root holding {split}/ and camera_intrinsics.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    src = data / a.split
    if not src.is_dir():
        raise SystemExit(f"no such split directory: {src}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    intr = json.loads((data / "camera_intrinsics.json").read_text())
    intr["sequences"] = {k: v for k, v in intr["sequences"].items() if k.startswith(f"{a.split}/")}
    (out / "camera_intrinsics.json").write_text(json.dumps(intr, indent=1))

    manifest = {"track": a.track, "split": a.split, "t_obs": T_OBS, "t_pred": T_PRED, "sequences": {}}
    for seq_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        frames = frames_of(seq_dir)
        imu = seq_dir / "imu.csv"
        dst = out / seq_dir.name
        dst.mkdir()
        if a.track == 1:
            for f in frames:
                place(f, dst / f.name)
            if imu.is_file():
                place(imu, dst / "imu.csv")
            manifest["sequences"][seq_dir.name] = {"frames": len(frames), "imu": imu.is_file()}
            continue
        starts = windows(len(frames))
        for s in starts:
            wdir = dst / f"{s:05d}"
            wdir.mkdir()
            for f in frames[s - 1:s - 1 + T_OBS]:
                place(f, wdir / f.name)
            if imu.is_file():
                imu_rows(imu, s, s + T_OBS - 1, wdir / "imu.csv")
        manifest["sequences"][seq_dir.name] = {"frames": len(frames), "imu": imu.is_file(), "windows": starts}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_win = sum(len(v.get("windows", [])) for v in manifest["sequences"].values())
    print(f"track {a.track}, {a.split}: {len(manifest['sequences'])} sequences"
          + (f", {n_win} windows" if a.track == 2 else "") + f" -> {out}")


if __name__ == "__main__":
    main()
