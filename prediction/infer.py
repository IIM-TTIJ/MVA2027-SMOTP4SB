"""Turn tracking output into Track 2 predictions, in submission format.

The training scripts report numbers; this is what produces a file you can
submit. It reads MOT tracking output, cuts the evaluation windows, predicts
T_PRED frames as K modes, and writes the six-column format the scorer expects:

    window_start, frame_id, track_id, mode_k, cx, cy

Predictors:

    stay        repeat the last observed position       (the floor)
    cv          constant velocity from the last 2 points
    kalman      Kalman filter, K sampled modes
    <a .pt>     a checkpoint from prediction.learned_models --save
                or prediction.train_sohota --save (one trained with --cmc
                also needs --imu-dir)
    <a PPT checkpoint>  PPT/training/training_ALL/<run>/model_<date>, a Stage III
                model from PPT patched by tools/patch_ppt.py (CUDA only)

`stay` and `cv` are deterministic, so their K modes are identical and minADE_K
collapses to ADE_1. That is expected; they exist as reference points.

Usage:
    python -m prediction.infer --tracks runs/track/ --out runs/pred/ --model stay
    python -m prediction.infer --tracks runs/track/ --out runs/pred/ \
        --model runs/lstm.pt --gpu 0
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

T_OBS_DEFAULT = 60
T_PRED_DEFAULT = 150
K_DEFAULT = 6


def load_tracks(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """MOT file -> {track_id: {frame_id: (cx, cy)}}, from box centres."""
    out: dict[int, dict[int, np.ndarray]] = collections.defaultdict(dict)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        if len(p) < 6:
            continue
        fid, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = (float(v) for v in p[2:6])
        out[tid][fid] = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    return dict(out)


def windows_for(tracks: dict[int, dict[int, np.ndarray]],
                t_obs: int, t_pred: int, first_frame: int = 1) -> list[int]:
    """Window start frames, matching evaluation.so_hota_tp.

    Windows start at `first_frame` and advance by t_pred while a whole
    t_obs + t_pred span still fits.

    **`first_frame` must be the sequence's first frame, not the first frame the
    tracker found anything in.** The scorer cuts its windows from the ground
    truth, which starts at frame 1; a detector that misses the opening frames
    would otherwise shift every window and every prediction would be keyed to a
    window the scorer never evaluates — scoring zero while looking correct. The
    default of 1 is the dataset's convention (images are `00001.jpg`).
    """
    frames = {f for t in tracks.values() for f in t}
    if not frames:
        return []
    hi = max(frames)
    return list(range(first_frame, hi - t_obs - t_pred + 2, t_pred))


def predict_stay(obs: np.ndarray, t_pred: int, k: int) -> np.ndarray:
    """(K, T_PRED, 2), every mode holding the last observed position."""
    return np.tile(obs[-1], (k, t_pred, 1))


def predict_cv(obs: np.ndarray, t_pred: int, k: int) -> np.ndarray:
    """Straight line through the last two observations.

    Not a lower bound, despite often being used as one: over 5 s it scores
    about twice as badly as `stay`, because a small error in the direction
    estimate becomes a large one at the end of the horizon.
    """
    v = obs[-1] - obs[-2] if len(obs) >= 2 else np.zeros(2)
    steps = np.arange(1, t_pred + 1)[:, None]
    return np.tile(obs[-1] + steps * v, (k, 1, 1))


def predict_kalman(obs_batch: np.ndarray, t_pred: int, k: int,
                   seed: int = 0) -> np.ndarray:
    """(B, T_OBS, 2) -> (B, K, T_PRED, 2). Batched: the filter already is.

    Settings match prediction/kalman.py's reported row, so the submission and
    the published baseline number come from the same configuration.
    """
    from prediction.kalman import kalman_filter, sampled_modes

    x, P = kalman_filter(obs_batch, q=1.0, r=10.0, damp=0.95)
    return sampled_modes(x, P, t_pred, k, damp=0.95, spread=1.0, seed=seed)


class TorchPredictor:
    """A checkpoint from prediction.learned_models or prediction.train_sohota --save.

    A checkpoint trained with a visibility head (train_sohota --vis) also says
    where each bird leaves: positions from the first frame whose visibility is
    below the saved threshold onwards come back as NaN, and the rows stop there.
    """

    def __init__(self, ckpt_path: Path, gpu: int = 0):
        import torch
        from prediction.learned_models import Predictor

        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.dev = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() and gpu >= 0 else "cpu"
        )
        self.vis_threshold = ck.get("vis_threshold", 0.5) if ck.get("vis") else None
        if self.vis_threshold is not None:
            from prediction.train_sohota import VisPredictor
            self.model = VisPredictor(ck["model"], ck["k"]).to(self.dev)
        else:
            self.model = Predictor(ck["model"], ck["k"], ck["mdn"]).to(self.dev)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.k, self.scale = ck["k"], ck["pos_scale"]
        self.t_obs, self.t_pred = ck["t_obs"], ck["t_pred"]
        self.cmc_mode = ck.get("cmc_mode")
        if self.cmc_mode not in (None, "input_only"):
            raise SystemExit(f"{ckpt_path} was trained with CMC mode {self.cmc_mode!r}; "
                             "only input_only can be run here")
        print(f"  loaded {ck['model']} (K={self.k}, val minADE "
              f"{ck.get('val_minADE', float('nan')):.2f}) on {self.dev}")

    def __call__(self, obs_batch: np.ndarray) -> np.ndarray:
        """(B, T_OBS, 2) absolute pixels -> (B, K, T_PRED, 2) absolute pixels.

        The model is trained on offsets from the last observed point, scaled;
        undo both so the caller only ever sees pixels.
        """
        origin = obs_batch[:, -1:, :]
        x = self.torch.tensor(
            ((obs_batch - origin) / self.scale).astype(np.float32)
        ).to(self.dev)
        with self.torch.no_grad():
            out = self.model(x)
        pred = out[0].cpu().numpy() * self.scale + origin[:, None]
        if self.vis_threshold is not None:
            visible = self.torch.sigmoid(out[3]).cpu().numpy() >= self.vis_threshold   # (B, T)
            # the track ends at the first frame predicted not visible; keep at least one
            gone = np.cumsum(~visible, axis=1) > 0
            gone[:, 0] = False
            pred[np.broadcast_to(gone[:, None, :], pred.shape[:3])] = np.nan
        return pred


class PPTPredictor:
    """A PPT Stage III checkpoint, run through PPT's own test model.

    PPT saves whole model objects, so its package has to be importable to
    unpickle them, and its forward pass allocates on CUDA unconditionally.
    """

    def __init__(self, ckpt_path: Path, gpu: int, t_obs: int, t_pred: int, k: int):
        import os
        import sys
        from types import SimpleNamespace

        import torch

        from prediction import paths

        os.environ.setdefault("PPT_PAST_LEN", str(t_obs))
        sys.path.insert(0, str(paths.PPT))
        from models.model_test_trajectory import Final_Model

        if not torch.cuda.is_available():
            raise SystemExit("PPT runs on CUDA only")
        self.torch = torch
        self.dev = torch.device(f"cuda:{max(gpu, 0)}")
        torch.cuda.set_device(self.dev)
        config = SimpleNamespace(cuda=True, gpu=max(gpu, 0), past_len=t_obs, future_len=t_pred,
                                 num_k=k, mode="ALL", dim_embedding_key=24)
        pretrained = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(self.dev)
        self.model = Final_Model(config, pretrained).to(self.dev).eval()
        print(f"  loaded PPT Stage III from {ckpt_path} on {self.dev}")

    def __call__(self, obs_batch: np.ndarray) -> np.ndarray:
        """(B, T_OBS, 2) absolute pixels -> (B, K, T_PRED, 2) absolute pixels."""
        t = self.torch
        origin = obs_batch[:, -1:, :]
        past = t.tensor((obs_batch - origin).astype(np.float32), device=self.dev)
        abs_past = t.tensor(obs_batch.astype(np.float32), device=self.dev)
        sse = t.tensor([[0, len(obs_batch)]], dtype=t.int32, device=self.dev)
        with t.no_grad():
            out = self.model(past, abs_past, sse, abs_past[:, -1]).cpu().numpy()
        return out + origin[:, None]


def build_predictor(name: str, k: int, t_pred: int, gpu: int, t_obs: int = T_OBS_DEFAULT):
    """Return (fn, batched) where fn maps observations to (K, T_PRED, 2)."""
    if Path(name).exists() and not name.endswith(".pt"):
        return PPTPredictor(Path(name), gpu, t_obs, t_pred, k), True
    if name.endswith(".pt") or Path(name).exists():
        return TorchPredictor(Path(name), gpu), True
    per_track = {"stay": predict_stay, "cv": predict_cv}
    if name in per_track:
        fn = per_track[name]
        return (lambda obs: fn(obs, t_pred, k)), False
    if name == "kalman":
        return (lambda batch: predict_kalman(batch, t_pred, k)), True
    raise SystemExit(
        f"unknown predictor {name!r}. Use stay, cv, kalman, or the path to a "
        f".pt checkpoint from prediction.learned_models --save."
    )


def observation_compensator(imu_dir: Path, seq: str, t_obs: int, cmc_config=None):
    """Camera motion compensation for a checkpoint trained with --cmc.

    Returns compensate(obs, window_start), mapping a (t_obs, 2) observation into
    the window's last observed frame from the IMU rows of the observed frames
    only, as training did. None when the sequence has no imu.csv: training left
    those sequences raw, so they are predicted raw here too.

    imu_dir is a dataset split directory such as SMOTP4SB_DATA/pub_test; its
    name is the split the camera intrinsics are looked up under. cmc_config is
    the JSON given to training with --cmc-config, if any ($SMOTP4SB_CMC_CONFIG
    otherwise).
    """
    from dataclasses import replace

    from cmc.config import CMCConfig
    from cmc.pipeline import compensate_observation_only, has_motion_source

    imu_dir = Path(imu_dir).resolve()
    cfg = replace(CMCConfig.load(cmc_config), enabled=True, imu_dir=imu_dir.parent,
                  split=imu_dir.name).with_mode("input_only").validate()
    if not has_motion_source(seq, cfg):
        return None
    return lambda obs, start: compensate_observation_only(obs, seq, start, t_obs, cfg)


def infer_sequence(tracks, predictor, batched, t_obs, t_pred, k,
                   first_frame: int = 1, compensate=None) -> list[str]:
    """Predict every (window, track) and format the rows.

    compensate, from observation_compensator, is applied to each observation
    before the model sees it; frames the track was not seen in stay out of it.
    """
    rows: list[str] = []
    for win in windows_for(tracks, t_obs, t_pred, first_frame):
        obs_frames = range(win, win + t_obs)
        pending_obs, pending_tid = [], []
        for tid, frames in sorted(tracks.items()):
            seen = sorted(f for f in frames if f in obs_frames)
            # Fewer than 5 observed frames and the scorer cannot associate this
            # track with the ground truth, so predicting it wastes rows.
            if len(seen) < 5:
                continue
            obs = np.stack([frames[f] for f in seen])
            if compensate is not None:
                full = np.full((t_obs, 2), np.nan)
                full[[f - win for f in seen]] = obs
                obs = compensate(full, win)[[f - win for f in seen]]
            if len(obs) < t_obs:
                # The model expects a fixed-length window; hold the first
                # position rather than extrapolating backwards into a guess.
                obs = np.vstack([np.tile(obs[0], (t_obs - len(obs), 1)), obs])
            pending_obs.append(obs)
            pending_tid.append(tid)

        if not pending_tid:
            continue
        if batched:
            modes_all = predictor(np.stack(pending_obs))
        else:
            modes_all = np.stack([predictor(o) for o in pending_obs])

        for tid, modes in zip(pending_tid, modes_all):
            for ki in range(k):
                for i in range(t_pred):
                    cx, cy = modes[ki, i]
                    if not (np.isfinite(cx) and np.isfinite(cy)):
                        break           # predicted to have left the view
                    rows.append(f"{win},{win + t_obs + i},{tid},{ki},"
                                f"{cx:.2f},{cy:.2f}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Predict Track 2 trajectories from tracking output")
    p.add_argument("--tracks", required=True,
                   help="directory of MOT files, one {seq}.txt per sequence")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default="stay",
                   help="stay | cv | kalman | path to a .pt checkpoint")
    p.add_argument("--t-obs", type=int, default=T_OBS_DEFAULT)
    p.add_argument("--t-pred", type=int, default=T_PRED_DEFAULT)
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--first-frame", type=int, default=1,
                   help="first frame of each sequence; windows are cut from "
                        "here, so it must match the ground truth, not the "
                        "first frame your tracker detected anything in")
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--imu-dir", default=None,
                   help="the dataset split directory holding {seq}/imu.csv, e.g. "
                        "SMOTP4SB_DATA/pub_test; read only by a checkpoint "
                        "trained with --cmc")
    p.add_argument("--cmc-config", default=None,
                   help="the JSON CMC config the checkpoint was trained with, if any")
    a = p.parse_args()

    tracks_dir, out_dir = Path(a.tracks), Path(a.out)
    files = sorted(tracks_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt tracking files in {tracks_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor, batched = build_predictor(a.model, a.k, a.t_pred, a.gpu)
    cmc = getattr(predictor, "cmc_mode", None) is not None
    if cmc and not a.imu_dir:
        raise SystemExit("this checkpoint was trained with --cmc, so its input must be "
                         "compensated too: pass --imu-dir, e.g. SMOTP4SB_DATA/pub_test")
    total = 0
    for f in files:
        tracks = load_tracks(f)
        compensate = (observation_compensator(Path(a.imu_dir), f.stem, a.t_obs, a.cmc_config)
                      if cmc else None)
        if cmc:
            print(f"  {f.stem}: {'IMU camera motion compensation' if compensate else 'no imu.csv, raw'}")
        rows = infer_sequence(
            tracks, predictor, batched, a.t_obs, a.t_pred, a.k, a.first_frame, compensate)
        (out_dir / f.name).write_text("\n".join(rows) + "\n" if rows else "")
        total += len(rows)
        print(f"  {f.stem}: "
              f"{len(windows_for(tracks, a.t_obs, a.t_pred, a.first_frame))} "
              f"window(s), {len(rows):,} rows")
    print(f"{len(files)} sequence(s), {total:,} predicted positions -> {out_dir}")


if __name__ == "__main__":
    main()
