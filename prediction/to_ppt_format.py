"""Convert SMOTP4SB trajectory windows into PPT's SocialDataset pickle format.

PPT loads `./{folder}/social_{scene}_{set}_{b_size}_{t_tresh}_{d_tresh}.pickle`
holding a `(traj, masks)` tuple, where:

  traj  : list of pre-formed batches, each (N_agents, T, >=4); columns 2:3 are x,y
  masks : list of (N, N) adjacency matrices; contiguous blocks of ones define
          which agents share a scene, which the loader turns into seq_start_end

We emit one agent per scene (identity masks). The Track 2 guide states that
single-trajectory prediction is sufficient for the baseline, and bird tracks in
this dataset rarely interact the way pedestrians do.

Usage:
    python -m prediction.to_ppt_format          # -> PPT/data/social_sdd_*.pickle
"""

import argparse
import pickle
from dataclasses import replace

import numpy as np

from prediction import paths
from prediction.build_dataset import gap_free_windows, load_tracks


def _compensator(cmc_cfg, t_obs):
    """Pick the compensation function once, rather than per window.

    The two take their arguments differently, so they are wrapped to a common
    (window, sequence, start_frame) call. Returns None when CMC is off, which is
    what the caller checks.
    """
    if cmc_cfg is None or not cmc_cfg.enabled:
        return None
    if cmc_cfg.mode == "input_only":
        from cmc.pipeline import compensate_observation_only
        return lambda w, seq, start: compensate_observation_only(
            w, seq, start, t_obs, cmc_cfg)
    from cmc.pipeline import compensate_window
    return lambda w, seq, start: compensate_window(
        w, seq, start, cmc_cfg, t_obs=t_obs)


def collect(t_obs, t_pred, val_seqs, stride, want_val, cmc_cfg=None,
            only_with_rotations=False):
    """Return (windows, meta).

    windows: (M, T, 2) pixel trajectories
    meta:    (M, 3) int64 rows of (sequence, track_id, first_frame)

    meta exists because PPT's pickle format keeps only coordinates, while camera
    motion compensation has to know which video and which frames a window came
    from in order to look up the camera's orientation. It is written alongside
    the pickle and stays in the same order.

    When cmc_cfg is enabled each window is compensated according to its mode:
    "full" transforms observation and target together, so the model both sees
    and predicts with the camera's motion removed; "input_only" transforms the
    observation alone and leaves the target in raw pixels.
    """
    win = t_obs + t_pred
    compensate = _compensator(cmc_cfg, t_obs)
    skipped = set()
    if only_with_rotations and compensate is not None:
        from cmc.pipeline import has_motion_source as has_rotation_source
    out, meta = [], []
    for (seq, tid), (frames, xy) in sorted(load_tracks().items()):
        if (seq in val_seqs) != want_val:
            continue
        # Opt-in: drop sequences that have no rotation source instead of
        # failing on them. This mixes nothing -- every window that is kept is
        # compensated -- but it changes the evaluation set, so the caller must
        # ask for it and the count is printed.
        if only_with_rotations and compensate is not None:
            if seq in skipped:
                continue
            if not has_rotation_source(seq, cmc_cfg):
                skipped.add(seq)
                continue
        # val is scored on non-overlapping windows; train packs them densely
        s_stride = win if want_val else stride
        for s in gap_free_windows(frames, win, s_stride):
            w = xy[s:s + win]
            start = int(frames[s])
            if compensate is not None:
                w = compensate(w, seq, start)
            out.append(w)
            meta.append((int(seq), int(tid), start))
    if skipped:
        print(f"    skipped {len(skipped)} sequence(s) with no rotation source: "
              f"{' '.join(sorted(skipped))}")
    return np.asarray(out, dtype=np.float32), np.asarray(meta, dtype=np.int64)


def to_batches(traj: np.ndarray, b_size: int):
    """Split (M, T, 2) into PPT batches of (N, T, 4) plus identity masks.

    PPT stacks the batch list with np.array(), so every batch must have the same
    agent count. A trailing partial batch makes the array inhomogeneous and the
    loader raises. We drop the remainder rather than pad, since padded rows would
    be scored as real trajectories.
    """
    n_full = len(traj) // b_size
    if n_full == 0:                      # fewer windows than one batch: keep them all
        n_full, b_size = 1, len(traj)
    dropped = len(traj) - n_full * b_size
    if dropped:
        print(f"    dropping {dropped} window(s) to keep batches uniform at {b_size}")
    traj = traj[:n_full * b_size]

    batches, masks = [], []
    for i in range(0, len(traj), b_size):
        chunk = traj[i:i + b_size]
        n, t, _ = chunk.shape
        # PPT reads columns 2:4 as (x, y); 0:2 are unused frame/id fields.
        padded = np.zeros((n, t, 4), dtype=np.float32)
        padded[:, :, 2:4] = chunk
        batches.append(padded)
        masks.append(np.eye(n, dtype=np.float32))
    return batches, masks


def main() -> None:
    p = argparse.ArgumentParser(description="SMOT4SB -> PPT pickle format")
    p.add_argument("--t-obs", type=int, default=60)
    p.add_argument("--t-pred", type=int, default=150)
    # stride=1 for training, to maximise data. stride=5 was an earlier
    # deviation and yields 13,139 windows instead of 65,165 --
    # it reproduces the wrong configuration, so it must not be the default.
    p.add_argument("--stride", type=int, default=1, help="training window stride")
    # PPT's `sdd` branch hardcodes folder='data', scene='sdd' and b_size 512/4096
    # (trainer_PPT.py:63-66), so these defaults are the only combination the
    # trainer will actually load. Renaming the scene means editing PPT.
    p.add_argument("--scene", type=str, default="sdd")
    p.add_argument("--train-b-size", type=int, default=512)
    p.add_argument("--test-b-size", type=int, default=4096)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    p.add_argument("--out", type=str, default=str(paths.PPT_DATA))
    p.add_argument("--cmc", action="store_true",
                   help="compensate camera motion (off by default)")
    p.add_argument("--cmc-mode", choices=("full", "input_only"), default=None,
                   help="full: compensate the whole window. input_only: "
                        "compensate the observation and leave the target in raw "
                        "pixels (default: whatever the config says)")
    p.add_argument("--cmc-config", default=None, help="JSON CMC config")
    p.add_argument("--only-with-rotations", action="store_true",
                   help="with --cmc: skip sequences that have no rotation source "
                        "instead of failing. Changes the evaluation set; the "
                        "skipped ids are printed.")
    a = p.parse_args()

    import json
    from cmc.config import CMCConfig
    cmc_cfg = CMCConfig.load(a.cmc_config)
    if a.cmc:
        cmc_cfg = replace(cmc_cfg, enabled=True)
    if a.cmc_mode:
        cmc_cfg = cmc_cfg.with_mode(a.cmc_mode)
    cmc_cfg = cmc_cfg.validate()
    if cmc_cfg.enabled:
        print(f"  CMC enabled: mode={cmc_cfg.mode}, reference={cmc_cfg.reference}, "
              f"camera motion from {cmc_cfg.source}")
    cmc_on = bool(cmc_cfg and cmc_cfg.enabled)
    val_seqs = set(json.load(open(paths.resolve(a.split)))["val_sequences"])
    out = paths.resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for set_name, want_val, b_size in (("train", False, a.train_b_size),
                                       ("test", True, a.test_b_size)):
        traj, meta = collect(a.t_obs, a.t_pred, val_seqs, a.stride, want_val, cmc_cfg,
                             only_with_rotations=a.only_with_rotations)
        batches, masks = to_batches(traj, b_size)
        # Name must match SocialDataset's f-string exactly (t_tresh/d_tresh unused here).
        name = out / f"social_{a.scene}_{set_name}_{b_size}_0_100.pickle"
        with open(name, "wb") as f:
            pickle.dump((batches, masks), f)
        # to_batches drops a trailing partial batch, so trim meta the same way
        # or the two fall out of step and every rotation lookup is wrong.
        meta_path = out / f"social_{a.scene}_{set_name}_{b_size}_0_100_meta.npz"
        np.savez_compressed(meta_path, meta=meta[:len(batches) * len(batches[0])],
                            cmc_enabled=cmc_on,
                            cmc_mode=cmc_cfg.mode if cmc_on else "none")
        print(f"  {set_name}: {len(traj)} windows -> {len(batches)} batches  {name}")


if __name__ == "__main__":
    main()
