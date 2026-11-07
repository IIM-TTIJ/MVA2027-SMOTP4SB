"""Score a trained PPT checkpoint and dump per-window errors.

PPT's own test entry point prints only dataset-level ADE/FDE, which is not enough
to compare it against another model: a paired significance test needs the error of
each window, not the mean. This script reuses PPT's exact forward path and metric
(trainer/test_final_trajectory.py) and additionally writes the per-window
distances, so the same checkpoint can be re-scored and bootstrapped without
re-training anything.

It also reports minADE_1 alongside minADE_6, as a mode-collapse diagnostic:
if minADE_6 is close to minADE_1, the modes have collapsed.

Run from the repo root:
    python tools/score_ppt.py --ckpt PPT/training/training_ALL/<run>/model_<date>
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PPT = ROOT / "PPT"


def main() -> None:
    p = argparse.ArgumentParser(description="Score a PPT checkpoint, per window")
    p.add_argument("--ckpt", required=True, help="path to a Stage III (ALL) checkpoint")
    p.add_argument("--past-len", type=int, default=60)
    p.add_argument("--future-len", type=int, default=150)
    p.add_argument("--num-k", type=int, default=6)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="runs/ppt_windows.npz")
    p.add_argument("--cmc", action="store_true",
                   help="undo camera motion on predictions before scoring")
    p.add_argument("--cmc-mode", choices=("full", "input_only"), default=None,
                   help="must match the mode the pickle was built with")
    p.add_argument("--cmc-config", default=None, help="JSON CMC config")
    p.add_argument("--extrapolation", default=None,
                   choices=["constant_omega", "uniform", "oracle"],
                   help="override how future camera rotation is predicted")
    a = p.parse_args()

    ckpt = Path(a.ckpt).resolve()
    out = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)

    # PPT resolves ./data and ./training against the working directory, and its
    # dataloader reads PPT_PAST_LEN at import time.
    os.environ["PPT_PAST_LEN"] = str(a.past_len)
    os.chdir(PPT)
    sys.path.insert(0, str(PPT))

    import torch
    from dataset_loader import SocialDataset, socialtraj_collate
    from models.model_test_trajectory import Final_Model
    from torch.utils.data import DataLoader

    config = SimpleNamespace(
        cuda=True, gpu=a.gpu, past_len=a.past_len, future_len=a.future_len,
        num_k=a.num_k, mode="ALL", dim_embedding_key=24,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)

    test_dataset = SocialDataset("data", set_name="test", b_size=4096,
                                 t_tresh=0, d_tresh=100, scene="sdd")
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=socialtraj_collate)

    # weights_only=False: PPT checkpoints contain full model objects, not just state_dict
    pretrained = torch.load(ckpt, map_location="cpu", weights_only=False).cuda()
    model = Final_Model(config, pretrained).cuda().eval()

    dev = torch.device("cuda")
    per_window = []          # (n, K, T) distances, matching PPT's own `distances`

    # Camera motion compensation. Under mode=full the predictions come back in
    # the reference frame's coordinates and must be mapped to raw pixels before
    # scoring, because that is the space the competition metric lives in --
    # scoring in compensated coordinates would grade an easier problem. Under
    # mode=input_only they are already raw and nothing has to be undone.
    #
    # cmc_cfg therefore stays None unless a mapping is actually needed: that is
    # exactly what the scoring loop below branches on.
    cmc_cfg = meta = None
    if a.cmc:
        sys.path.insert(0, str(ROOT))
        from cmc.config import CMCConfig
        from cmc.pipeline import decompensate_batch
        from dataclasses import replace as _replace
        cfg = _replace(CMCConfig.load(a.cmc_config), enabled=True)
        if a.cmc_mode:
            cfg = cfg.with_mode(a.cmc_mode)
        if a.extrapolation:
            cfg = _replace(cfg, extrapolation=a.extrapolation)
        cfg = cfg.validate()

        meta_path = PPT / "data" / "social_sdd_test_4096_0_100_meta.npz"
        if not meta_path.exists():
            raise SystemExit(f"--cmc needs the window metadata at {meta_path}\n"
                             f"  rebuild it: python -m prediction.to_ppt_format")
        meta_file = np.load(meta_path)
        if not bool(meta_file.get("cmc_enabled", False)):
            raise SystemExit("--cmc passed but the pickle was built without --cmc;\n"
                             "  rebuild with: python -m prediction.to_ppt_format --cmc")
        # The two modes leave the trajectories in different coordinate spaces, and
        # scoring the wrong one shifts every prediction by the accumulated rotation
        # without failing. The sidecar records which was built so the mismatch is
        # impossible rather than merely unlikely. Sidecars written before the mode
        # existed can only be full CMC.
        built = str(meta_file["cmc_mode"]) if "cmc_mode" in meta_file else "full"
        if built != cfg.mode:
            raise SystemExit(
                f"the pickle was built with CMC mode {built!r} but scoring asked "
                f"for {cfg.mode!r}.\n"
                f"  These are different coordinate spaces. Rebuild with:\n"
                f"    python -m prediction.to_ppt_format --cmc --cmc-mode {cfg.mode}")
        meta = meta_file["meta"]

        if cfg.mode == "input_only":
            # Nothing to undo. compensate_observation_only references the LAST
            # observed frame, where the homography is exactly the identity, so
            # the anchor PPT predicts from keeps its true pixel position, and the
            # target frames were never compensated at all. output + anchor is
            # therefore already in raw pixels and the pickle's future is raw, so
            # the plain scoring path below is the correct one -- leaving cmc_cfg
            # as None is the whole implementation.
            print("CMC on: mode=input_only -- predictions are already in raw "
                  "pixels, no decompensation")
        else:
            cmc_cfg = cfg
            print(f"CMC on: mode=full, extrapolation={cfg.extrapolation}, "
                  f"rotations from {cfg.rotation_dir}")

    offset = 0
    with torch.no_grad():
        for trajectory, mask, initial_pos, seq_start_end in loader:
            trajectory = torch.FloatTensor(trajectory).to(dev)
            initial_pos = torch.FloatTensor(initial_pos).to(dev)

            # Identical to test_final_trajectory.evaluate()
            traj_norm = trajectory - trajectory[:, a.past_len - 1:a.past_len, :]
            x = traj_norm[:, :a.past_len, :]
            abs_past = trajectory[:, :a.past_len, :]
            initial_pose = trajectory[:, a.past_len - 1:a.past_len, :]

            output = model(x, abs_past, seq_start_end, initial_pose).data

            if cmc_cfg is None:
                future_rep = traj_norm[:, a.past_len:, :].unsqueeze(1).repeat(1, a.num_k, 1, 1)
                per_window.append(torch.norm(output - future_rep, dim=3).cpu().numpy())
            else:
                # PPT predicts relative to the last observed point; restore the
                # absolute (still compensated) pixel positions before undoing.
                abs_modes = (output + initial_pose.unsqueeze(1)).cpu().numpy().astype(np.float64)
                chunk = meta[offset:offset + len(abs_modes)]
                raw_modes, raw_gt = decompensate_batch(abs_modes, chunk, a.past_len, cmc_cfg)
                per_window.append(np.linalg.norm(raw_modes - raw_gt[:, None], axis=3))
            if cmc_cfg is not None:
                offset += len(output)

    d = np.concatenate(per_window, axis=0)           # (N, K, T)
    ade_per_mode = d.mean(axis=2)                    # (N, K)
    fde_per_mode = d[:, :, -1]                       # (N, K)

    ade_k = ade_per_mode.min(axis=1)                 # minADE_K, per window
    fde_k = fde_per_mode.min(axis=1)
    # minADE_1: one hypothesis instead of K. Mode order is arbitrary (PPT does not
    # rank its destinations), so the mean over modes is the honest "one random
    # hypothesis" reference; mode 0 is reported too in case they diverge.
    ade_1_mean = ade_per_mode.mean(axis=1)
    ade_1_mode0 = ade_per_mode[:, 0]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, distances=d, ade_k=ade_k, fde_k=fde_k,
                        ade_1_mean=ade_1_mean, ade_1_mode0=ade_1_mode0)

    n = len(d)
    print(f"checkpoint: {ckpt}")
    print(f"windows:    {n}   K={a.num_k}   {a.past_len} -> {a.future_len}\n")
    print(f"  minADE_{a.num_k}        {ade_k.mean():8.2f}")
    print(f"  minFDE_{a.num_k}        {fde_k.mean():8.2f}")
    print(f"  minADE_1 (mean over modes) {ade_1_mean.mean():8.2f}   "
          f"ratio {ade_k.mean()/ade_1_mean.mean():.3f}")
    print(f"  minADE_1 (mode 0)          {ade_1_mode0.mean():8.2f}   "
          f"ratio {ade_k.mean()/ade_1_mode0.mean():.3f}")
    print(f"\n  per-window errors -> {out}")


if __name__ == "__main__":
    main()
