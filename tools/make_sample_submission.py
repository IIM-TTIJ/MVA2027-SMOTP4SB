"""Build a valid Track 2 submission, to show the format and to test the scorer.

The `stay` baseline: every mode repeats the last observed position. It is not a
good predictor -- it is the floor a real model must beat -- but it produces a
structurally correct archive, which is what this is for.

Run it against GT to see the shape of a submission, or against your own tracker
output to check your pipeline end to end before uploading.

Usage:
    # from GT (train split), for one sequence
    python tools/make_sample_submission.py --gt SMOT4SB/annotations/train.json \
        --sequences 0001 0002 --out sample_submission.zip

    # from your own tracking output, one MOT file per sequence
    python tools/make_sample_submission.py --tracking-dir runs/track/ \
        --out submission.zip
"""

import argparse
import collections
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T_OBS, T_PRED, K = 60, 150, 6


def tracks_from_coco(ann_file: Path, sequences: set[str] | None) -> dict:
    """seq -> track_id -> {frame_id: (cx, cy, w, h)} from a COCO/MOT json."""
    data = json.loads(Path(ann_file).read_text())
    images = {i["id"]: i for i in data["images"]}
    out: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for a in data["annotations"]:
        im = images[a["image_id"]]
        seq = im["file_name"].split("/")[0]
        if sequences and seq not in sequences:
            continue
        x, y, w, h = a["bbox"]
        out[seq][a["track_id"]][im["frame_id"]] = (x + w / 2, y + h / 2, w, h)
    return out


def tracks_from_mot_dir(d: Path, sequences: set[str] | None) -> dict:
    """Same shape, from one MOT text file per sequence."""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in sorted(Path(d).glob("*.txt")):
        seq = f.stem
        if sequences and seq not in sequences:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            p = line.split(",")
            fid, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = (float(v) for v in p[2:6])
            out[seq][tid][fid] = (x + w / 2, y + h / 2, w, h)
    return out


def windows_for(seq_tracks: dict) -> list[int]:
    """Window starts, matching evaluation.so_hota_tp.define_evaluation_windows.

    Windows begin at frame 1 and advance by T_PRED, for as long as a whole
    T_OBS + T_PRED span fits.
    """
    frames = {f for t in seq_tracks.values() for f in t}
    if not frames:
        return []
    return list(range(1, max(frames) - T_OBS - T_PRED + 2, T_PRED))


def build(tracks: dict, out_zip: Path) -> None:
    n_rows = n_win = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "prediction").mkdir()
        (tmp / "tracking").mkdir()

        for seq, seq_tracks in sorted(tracks.items()):
            pred_lines: list[str] = []
            track_lines: list[str] = []

            for win in windows_for(seq_tracks):
                obs = range(win, win + T_OBS)
                n_win += 1
                for tid, frames in sorted(seq_tracks.items()):
                    seen = sorted(f for f in frames if f in obs)
                    if len(seen) < 5:
                        # Too short to associate; a real tracker would not
                        # report it either.
                        continue
                    for f in seen:
                        cx, cy, w, h = frames[f]
                        track_lines.append(
                            f"{f},{tid},{cx - w / 2:.2f},{cy - h / 2:.2f},"
                            f"{w:.2f},{h:.2f},1.0"
                        )
                    cx, cy, _, _ = frames[seen[-1]]
                    for k in range(K):
                        for i in range(T_PRED):
                            pred_lines.append(
                                f"{win},{win + T_OBS + i},{tid},{k},"
                                f"{cx:.2f},{cy:.2f}"
                            )
                            n_rows += 1

            if not pred_lines:
                continue
            # A track is observed by several windows, so tracking rows repeat.
            seen_rows = dict.fromkeys(track_lines)
            (tmp / "prediction" / f"{seq}.txt").write_text(
                "\n".join(pred_lines) + "\n"
            )
            (tmp / "tracking" / f"{seq}.txt").write_text(
                "\n".join(seen_rows) + "\n"
            )

        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(tmp.rglob("*.txt")):
                z.write(f, f.relative_to(tmp))

    print(f"{out_zip}: {n_win} window(s), {n_rows:,} predicted positions")
    print("Validate it with:")
    print(f"  python tools/validate_submission.py {out_zip}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a sample Track 2 submission")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt", help="COCO/MOT annotations json (train split)")
    src.add_argument(
        "--tracking-dir", help="directory of MOT files, one per sequence"
    )
    p.add_argument(
        "--sequences", nargs="*", default=None,
        help="limit to these sequence ids (default: all)",
    )
    p.add_argument("--out", default="sample_submission.zip")
    a = p.parse_args()

    seqs = set(a.sequences) if a.sequences else None
    tracks = (
        tracks_from_coco(Path(a.gt), seqs)
        if a.gt
        else tracks_from_mot_dir(Path(a.tracking_dir), seqs)
    )
    if not tracks:
        sys.exit("no tracks found — check --sequences and the input path")
    build(tracks, Path(a.out))


if __name__ == "__main__":
    main()
