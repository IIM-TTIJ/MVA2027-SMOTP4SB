"""Masked window builder: separates *gaps* from *disappearance*.

A companion to build_dataset.py, not a replacement -- it imports that module's
load_tracks() and gap_free_windows(). build_dataset.py produces the standard
set where all 210 frames are present; this one produces the set the task
actually scores (docs/track2.md), where the future may end early.

The original PPT recipe says "exclude windows that contain detection gaps; do not
interpolate", and the task definition says 43% of tracks disappear during the prediction
period and that this is the point of the benchmark. Taken together as originally
implemented they cancel out: requiring all 210 frames present discards every
disappearing track, so the baseline never sees the majority case.

The two are different phenomena and need different handling:

  interior gap  - a frame missing mid-track (occlusion, missed detection).
                  Excluded, as intended; interpolating would invent data.
  truncation    - the track simply ends (bird leaves frame). NOT a gap. This is
                  disappearance, and the task scores it.

So: the 60-frame observation must be gap-free (you cannot predict from a broken
input), while the future may end early. Frames after the end are marked invalid
and scored over GT-present frames only -- which is what section 7 says the metric
already does.
"""

import argparse
import json

import numpy as np

from prediction import paths
from prediction.build_dataset import gap_free_windows, load_tracks

T_OBS, T_PRED = 60, 150


def build(t_obs, t_pred, stride, split_file, want_val, cmc_cfg=None):
    """Return (obs, fut, mask). mask[i, t] is True where GT exists.

    With cmc_cfg enabled, each observation from a sequence that has imu.csv is
    compensated into its last frame (input_only; the future stays raw), and
    sequences without one are left raw.

    NOTE ON STRIDE -- it does not mean the same thing here as in build_dataset.py.
    There, stride steps over windows with a gap-free 210 frames; here it steps over
    windows with a gap-free 60-frame *observation*, since the future is allowed to
    end early. The eligible-start lists therefore differ, and so does what the
    greedy stepping in gap_free_windows picks.

    Consequence: filtering this module's output to complete futures reproduces
    build_dataset.py exactly at stride 1 (65,165 train) and stride 30 (182 val),
    but NOT at stride 210, where it yields 43 windows against build_dataset.py's
    44 -- track 0006/34 is dropped because the stepping passes its only qualifying
    start. Build the official n=44 evaluation set with build_dataset.py, never by
    striding this module at 210.
    """
    val = set(json.load(open(paths.resolve(split_file)))["val_sequences"])
    if cmc_cfg is not None and cmc_cfg.enabled:
        from cmc.pipeline import compensate_observation_only, has_motion_source
    O, F, M = [], [], []
    for (seq, _), (frames, xy) in sorted(load_tracks().items()):
        if (seq in val) != want_val:
            continue
        compensate = (cmc_cfg is not None and cmc_cfg.enabled
                      and has_motion_source(seq, cmc_cfg))
        pos = dict(zip(frames.tolist(), xy))
        for i in gap_free_windows(frames, t_obs, stride):
            last = int(frames[i + t_obs - 1])
            fut = np.zeros((t_pred, 2), dtype=np.float32)
            msk = np.zeros(t_pred, dtype=bool)
            for j in range(t_pred):
                f = last + 1 + j
                if f not in pos:
                    break            # truncation: everything after is absent
                fut[j] = pos[f]; msk[j] = True
            if msk.sum() < 2:        # need at least a couple of GT frames to score
                continue
            obs = xy[i:i + t_obs]
            if compensate:
                obs = compensate_observation_only(obs, seq, int(frames[i]), t_obs, cmc_cfg)
            O.append(obs); F.append(fut); M.append(msk)
    return (np.asarray(O, np.float32), np.asarray(F, np.float32), np.asarray(M))


def masked_metrics(modes, fut, mask, thr=48.0):
    """minADE_K / minFDE_K / MissRate over GT-present frames only.

    modes: (B, K, T, 2); mask: (B, T). The final displacement is taken at each
    window's own last valid frame, not at frame 150.
    """
    d = np.linalg.norm(modes - fut[:, None], axis=-1)          # (B,K,T)
    m = mask[:, None, :]
    ade = (d * m).sum(2) / np.maximum(m.sum(2), 1)             # masked mean
    last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)  # last valid index
    fde = d[np.arange(len(d))[:, None], np.arange(d.shape[1])[None, :], last[:, None]]
    b = ade.argmin(1); rows = np.arange(len(d))
    return dict(minADE=float(ade.min(1).mean()), minFDE=float(fde.min(1).mean()),
                MR=float((fde[rows, b] > thr).mean()), n=len(d))


def main():
    p = argparse.ArgumentParser(description="Disappearance-aware window builder")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()
    o, f, m = build(T_OBS, T_PRED, a.stride, a.split, True)
    full = m.all(1)
    print(f"  val windows (obs gap-free, future may truncate): {len(o)}")
    print(f"    future complete      : {full.sum()}  ({full.mean()*100:.1f}%)")
    print(f"    bird disappears      : {(~full).sum()}  ({(~full).mean()*100:.1f}%)")
    print(f"    median GT frames when disappearing: {int(np.median(m[~full].sum(1)))}/{T_PRED}")


if __name__ == "__main__":
    main()
