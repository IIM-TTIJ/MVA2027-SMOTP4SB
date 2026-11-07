"""Package tracking and prediction results into submission archives.

One archive per track, because the two are separate CodaBench competitions:

    track1_submission.zip    {seq}.txt                  MOT tracking
    track2_submission.zip    prediction/{seq}.txt       6-column predictions
                             tracking/{seq}.txt         observed frames, for
                                                        matching your ids to GT

The Track 2 archive carries the tracking output as well. Track ids are yours to
choose and mean nothing to the scorer, so without it nothing can be matched to
the ground truth and nothing can be scored.

Validation runs before the archive is written, so a malformed submission fails
here rather than as a silent zero on the leaderboard.

Usage:
    # both tracks
    python tools/make_submission.py --tracks runs/track/ --pred runs/pred/ \
        --out runs/submission/

    # tracking only
    python tools/make_submission.py --tracks runs/track/ --out runs/submission/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def zip_dir(src: Path, out: Path, prefix: str = "") -> int:
    """Zip every .txt in `src` at `prefix/name`. Returns the file count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.txt"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"{prefix}{f.name}" if prefix else f.name)
    return len(files)


def build_track1(tracks: Path, out_dir: Path) -> Path:
    out = out_dir / "track1_submission.zip"
    n = zip_dir(tracks, out)
    if n == 0:
        raise SystemExit(f"no .txt files in {tracks}")
    print(f"Track 1: {n} sequence(s) -> {out}")
    return out


def build_track2(tracks: Path, pred: Path, out_dir: Path) -> Path:
    pred_files = sorted(pred.glob("*.txt"))
    if not pred_files:
        raise SystemExit(f"no .txt files in {pred}")

    missing = [f.name for f in pred_files if not (tracks / f.name).exists()]
    if missing:
        raise SystemExit(
            f"predictions exist for {missing[:5]} but the matching tracking "
            f"file does not. Both are required: ids are matched to ground "
            f"truth through the tracking output."
        )

    out = out_dir / "track2_submission.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pred = n_trk = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in pred_files:
            z.write(f, f"prediction/{f.name}")
            n_pred += 1
            z.write(tracks / f.name, f"tracking/{f.name}")
            n_trk += 1
    print(f"Track 2: {n_pred} prediction + {n_trk} tracking file(s) -> {out}")
    return out


def validate(archive: Path) -> bool:
    """Run the Track 2 validator; True if the archive is well-formed."""
    script = ROOT / "tools" / "validate_submission.py"
    if not script.exists():
        print("  (validator not found, skipping)")
        return True
    r = subprocess.run(
        [sys.executable, str(script), str(archive)],
        capture_output=True, text=True,
    )
    print(r.stdout.rstrip())
    if r.stderr.strip():
        print(r.stderr.rstrip(), file=sys.stderr)
    return r.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Package results into CodaBench submission archives")
    p.add_argument("--tracks", required=True,
                   help="directory of MOT tracking files, one per sequence")
    p.add_argument("--pred", default=None,
                   help="directory of Track 2 predictions (omit for Track 1 only)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--skip-validation", action="store_true",
                   help="write the archive even if validation fails")
    a = p.parse_args()

    tracks, out_dir = Path(a.tracks), Path(a.out)
    if not tracks.is_dir():
        raise SystemExit(f"no such directory: {tracks}")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_track1(tracks, out_dir)

    if a.pred:
        pred = Path(a.pred)
        if not pred.is_dir():
            raise SystemExit(f"no such directory: {pred}")
        archive = build_track2(tracks, pred, out_dir)
        print("\nValidating the Track 2 archive:")
        if not validate(archive) and not a.skip_validation:
            # Leave the archive in place so its contents can be inspected, but
            # exit non-zero: uploading it would score zero without saying why.
            raise SystemExit(
                "\nValidation failed. Fix the errors above and rebuild; the "
                "archive was written but should not be uploaded as is. "
                "Pass --skip-validation to override."
            )

    print(f"\nArchives in {out_dir}")
    print("  Track 1 and Track 2 are separate CodaBench competitions; upload "
          "each to its own.")


if __name__ == "__main__":
    main()
