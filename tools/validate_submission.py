"""Check a Track 2 submission before uploading it.

The scoring server cannot tell a deliberate choice from a mistake. Stopping a
track early means "I predict it disappears here" and is scored as such; a
missing mode is simply a mode that scores nothing.

A track may be absent from any frames of the horizon: it may stop, start late
or resume after a gap, and each frame is scored on its own. Both look like valid
submissions. This script flags the shapes that are usually accidents, so a
malformed upload is caught locally rather than as a low score.

Usage:
    python tools/validate_submission.py submission.zip
    python tools/validate_submission.py path/to/unzipped/dir
"""

import argparse
import collections
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T_OBS, T_PRED, K = 60, 150, 6


class Report:
    """Errors block the upload; warnings are usually, but not always, bugs."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def print(self) -> bool:
        for m in self.notes:
            print(f"  {m}")
        if self.warnings:
            print(f"\n{len(self.warnings)} warning(s):")
            for m in self.warnings[:40]:
                print(f"  ! {m}")
            if len(self.warnings) > 40:
                print(f"  ... and {len(self.warnings) - 40} more")
        if self.errors:
            print(f"\n{len(self.errors)} error(s):")
            for m in self.errors[:40]:
                print(f"  x {m}")
            if len(self.errors) > 40:
                print(f"  ... and {len(self.errors) - 40} more")
            print("\nFAILED — fix the errors above before submitting.")
            return False
        print("\nOK — the submission is well-formed.")
        if self.warnings:
            print("Review the warnings: they are legal but often unintended.")
        return True


def _parse_prediction(path: Path, rep: Report) -> dict:
    """Read one prediction file into window -> track -> mode -> {frame: (x, y)}."""
    data: dict = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 6:
            rep.error(
                f"{path.name}:{n}: expected 6 comma-separated columns "
                f"(window_start, frame_id, track_id, mode_k, cx, cy), got {len(parts)}"
            )
            continue
        try:
            win, fid, tid, k = (int(float(p)) for p in parts[:4])
            cx, cy = (float(p) for p in parts[4:])
        except ValueError:
            rep.error(f"{path.name}:{n}: non-numeric field")
            continue
        if not 0 <= k < K:
            rep.error(f"{path.name}:{n}: mode_k={k} outside 0..{K - 1}")
            continue
        slot = data.setdefault(win, {}).setdefault(tid, {}).setdefault(k, {})
        if fid in slot:
            rep.error(
                f"{path.name}:{n}: duplicate row for "
                f"window {win}, track {tid}, mode {k}, frame {fid}"
            )
        slot[fid] = (cx, cy)
    return data


def _parse_tracking(path: Path, rep: Report) -> dict:
    """Read one tracking file into track -> {frame: (cx, cy)}; MOT columns."""
    data: dict = collections.defaultdict(dict)
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            rep.error(
                f"tracking/{path.name}:{n}: expected at least 6 MOT columns "
                f"(frame_id, track_id, x, y, w, h), got {len(parts)}"
            )
            continue
        try:
            fid, tid = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = (float(p) for p in parts[2:6])
        except ValueError:
            rep.error(f"tracking/{path.name}:{n}: non-numeric field")
            continue
        data[tid][fid] = (x + w / 2.0, y + h / 2.0)
    return data


def _check_sequence(
    seq: str, pred: dict, track: dict | None, rep: Report
) -> None:
    for win, tracks in sorted(pred.items()):
        lo, hi = win + T_OBS, win + T_OBS + T_PRED - 1
        for tid, modes in sorted(tracks.items()):
            missing = sorted(set(range(K)) - set(modes))
            if missing:
                rep.warn(
                    f"{seq} window {win} track {tid}: only {len(modes)} of {K} "
                    f"modes (missing {missing}). Absent modes score nothing."
                )
            for k, frames in sorted(modes.items()):
                fids = sorted(frames)
                out = [f for f in fids if not lo <= f <= hi]
                if out:
                    rep.error(
                        f"{seq} window {win} track {tid} mode {k}: "
                        f"{len(out)} frame(s) outside the horizon {lo}..{hi} "
                        f"(e.g. {out[:3]}). Predict only the {T_PRED} frames "
                        f"after the {T_OBS} observed ones."
                    )
                    continue
                # Absent frames are legal anywhere in the horizon: stopping,
                # starting late and resuming after a gap are all predictions
                # that the bird is out of view there, and each frame is
                # scored on its own.
            present = {k: frozenset(v) for k, v in modes.items()}
            if len(set(present.values())) > 1:
                rep.warn(
                    f"{seq} window {win} track {tid}: modes cover different "
                    f"frames ({ {k: len(v) for k, v in present.items()} }). Each "
                    f"mode is a separate hypothesis, so this is legal, but it "
                    f"is usually a bug."
                )
            elif modes and len(next(iter(present.values()))) < T_PRED:
                rep.note(
                    f"{seq} window {win} track {tid}: "
                    f"{len(next(iter(present.values())))}/{T_PRED} frames "
                    f"- the rest are read as the bird being out of view."
                )
        if track is not None:
            unknown = sorted(set(tracks) - set(track))
            if unknown:
                rep.error(
                    f"{seq} window {win}: track id(s) {unknown[:5]} appear in "
                    f"prediction but not in tracking/{seq}.txt. Ids are matched "
                    f"to ground truth through the tracking file, so a predicted "
                    f"track absent from it can never be scored."
                )
            for tid in set(tracks) & set(track):
                seen = [f for f in track[tid] if win <= f < win + T_OBS]
                if len(seen) < 5:
                    rep.warn(
                        f"{seq} window {win} track {tid}: only {len(seen)} "
                        f"observed frame(s) in tracking/. Association needs at "
                        f"least 5 to match this track to ground truth."
                    )


def validate(root: Path) -> bool:
    rep = Report()
    pred_dir, track_dir = root / "prediction", root / "tracking"

    if not pred_dir.is_dir():
        # Tolerate a zip with one extra top-level folder.
        inner = [d for d in root.iterdir() if (d / "prediction").is_dir()]
        if len(inner) == 1:
            return validate(inner[0])
        rep.error(
            "no prediction/ directory. Expected prediction/{seq}.txt and "
            "tracking/{seq}.txt at the top level of the archive."
        )
        return rep.print()

    pred_files = sorted(pred_dir.glob("*.txt"))
    if not pred_files:
        rep.error("prediction/ contains no .txt files")
        return rep.print()

    if not track_dir.is_dir():
        rep.error(
            "no tracking/ directory. It carries your tracker's output over the "
            "observed frames, which is how your track ids are matched to ground "
            "truth. Without it nothing can be scored."
        )

    rep.note(f"{len(pred_files)} sequence file(s) in prediction/")
    total_rows = 0
    for pf in pred_files:
        seq = pf.stem
        pred = _parse_prediction(pf, rep)
        total_rows += sum(
            len(fr) for t in pred.values() for m in t.values() for fr in m.values()
        )
        track = None
        if track_dir.is_dir():
            tf = track_dir / f"{seq}.txt"
            if tf.exists():
                track = _parse_tracking(tf, rep)
            else:
                rep.error(f"tracking/{seq}.txt is missing")
        _check_sequence(seq, pred, track, rep)

    rep.note(f"{total_rows:,} predicted positions total")
    return rep.print()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a Track 2 submission")
    p.add_argument("path", help="submission .zip or an unzipped directory")
    a = p.parse_args()

    src = Path(a.path)
    if not src.exists():
        sys.exit(f"not found: {src}")

    if src.is_file():
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with zipfile.ZipFile(src) as z:
                    z.extractall(tmp)
            except zipfile.BadZipFile:
                sys.exit(f"{src} is not a readable zip archive")
            ok = validate(Path(tmp))
    else:
        ok = validate(src)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
