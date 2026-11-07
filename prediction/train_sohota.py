"""Train the Track 2 LSTM with a loss shaped like SO-HOTA-TP's localization term.

    python -m prediction.train_sohota --loss dotd --lam 0.05 --gpu 0 \
        --split data/traj/split.json --save runs/lstm_dotd.pt

SO-HOTA-TP picks, per track and window, the mode with the highest mean DotD
over the frames where the bird is visible, and a predicted point counts only
when DotD = exp(-d / S) clears a threshold alpha in 0.05..0.95. Averaged over
those thresholds the match indicator is close to DotD itself, so the loss here
is 1 - exp(-d / S) averaged over visible frames, winner-takes-all over K modes
chosen by that same quantity. Windows whose future ends early (the bird leaves)
are kept and scored on their visible frames, as the metric does.

--loss l2 keeps the old objective (best-of-K mean L2) on the same masked
windows, to separate the effect of the data from the effect of the loss.

--cmc compensates the observation of sequences that have imu.csv into its last
frame (input_only, docs/cmc.md); run the checkpoint with prediction.infer
--imu-dir so the input is compensated the same way.
The checkpoint has the format prediction/infer.py loads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prediction.build_dataset_masked import build
from prediction.learned_models import POS_SCALE, T_OBS, T_PRED, Predictor

S_DOTD = 24.15  # the dataset's mean box size, as in the scorer


class VisPredictor(Predictor):
    """Predictor plus a per-frame visibility head: is the bird still in view at t?

    SO-HOTA-TP counts positions emitted after the bird has gone as false
    positives, so a submission should stop a track where the bird leaves. The
    head is trained against the visible-frame mask; at inference the track is
    cut at the first frame whose visibility falls below a threshold.
    """

    def __init__(self, kind, k, d=128):
        super().__init__(kind, k, False, d)
        self.vis = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, T_PRED))

    def forward(self, x):
        h = self.enc(x)
        mu = self.head(h)[:, :self.k * T_PRED * 2].view(len(x), self.k, T_PRED, 2)
        return mu, None, None, self.vis(h)


def tensors(o, f, m, dev):
    origin = o[:, -1:, :]
    x = torch.tensor((o - origin) / POS_SCALE, device=dev)
    y = torch.tensor((f - origin) / POS_SCALE, device=dev)
    return x, y, torch.tensor(m, device=dev, dtype=torch.float32)


def per_mode(mu, y, m, kind):
    """(B,K) loss per mode over visible frames."""
    d = torch.norm(mu - y[:, None], dim=-1)                     # (B,K,T), scaled units
    w = m[:, None] / m.sum(1).clamp(min=1)[:, None, None]
    if kind == "l2":
        return (d * w).sum(-1)
    return ((1 - torch.exp(-d * POS_SCALE / S_DOTD)) * w).sum(-1)


def loss_fn(mu, y, m, kind, lam):
    dotd = per_mode(mu, y, m, "dotd")
    if kind == "l2":
        return per_mode(mu, y, m, "l2").min(1).values.mean()
    best = dotd.argmin(1)                                       # the mode the scorer would keep
    out = dotd.gather(1, best[:, None]).mean()
    if lam > 0:
        out = out + lam * per_mode(mu, y, m, "l2").gather(1, best[:, None]).mean()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loss", choices=("l2", "dotd"), default="dotd")
    p.add_argument("--lam", type=float, default=0.05, help="weight of the L2 term with --loss dotd")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", required=True)
    p.add_argument("--save", required=True)
    p.add_argument("--vis", type=float, default=0.0,
                   help="weight of a visibility head (BCE on the visible-frame mask); 0 = no head")
    p.add_argument("--vis-threshold", type=float, default=0.5,
                   help="cut a predicted track where visibility first drops below this")
    p.add_argument("--cmc", action="store_true",
                   help="compensate camera motion in the observation (input_only)")
    p.add_argument("--cmc-config", default=None, help="JSON CMC config")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    cmc_cfg = None
    if a.cmc:
        from dataclasses import replace
        from cmc.config import CMCConfig
        cmc_cfg = replace(CMCConfig.load(a.cmc_config), enabled=True).with_mode("input_only").validate()
        print(f"CMC on: input_only, model {cmc_cfg.egomotion_model}", flush=True)
    o, f, m = build(T_OBS, T_PRED, a.stride, a.split, False, cmc_cfg)
    vo, vf, vm = build(T_OBS, T_PRED, 30, a.split, True, cmc_cfg)
    print(f"train {len(o)} windows ({(~m.all(1)).mean() * 100:.0f}% end early), "
          f"val {len(vo)} ({(~vm.all(1)).mean() * 100:.0f}% end early)", flush=True)
    X, Y, M = tensors(o, f, m, dev)
    VX, VY, VM = tensors(vo, vf, vm, dev)

    model = (VisPredictor("lstm", a.k) if a.vis > 0 else Predictor("lstm", a.k, False)).to(dev)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for i in range(0, len(X), a.batch):
            idx = perm[i:i + a.batch]
            out = model(X[idx])
            loss = loss_fn(out[0], Y[idx], M[idx], a.loss, a.lam)
            if a.vis > 0:
                loss = loss + a.vis * bce(out[3], M[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            mu = model(VX)[0]
            score = (1 - per_mode(mu, VY, VM, "dotd")).max(1).values.mean().item()   # mean best-mode DotD
            ade = (per_mode(mu, VY, VM, "l2") * POS_SCALE).min(1).values.mean().item()
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1:3}  loss {tot / len(X):.4f}  val best-mode DotD {score:.4f}  masked minADE {ade:7.2f}", flush=True)

    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model": "lstm", "k": a.k, "mdn": False,
                "t_obs": T_OBS, "t_pred": T_PRED, "pos_scale": POS_SCALE,
                "cmc_mode": "input_only" if cmc_cfg else None,
                "val_minADE": float("nan"), "val_dotd": best, "seed": a.seed,
                "loss": a.loss, "lam": a.lam, "vis": a.vis > 0, "vis_threshold": a.vis_threshold}, a.save)
    print(f"saved {a.save}  (best val best-mode DotD {best:.4f})")


if __name__ == "__main__":
    main()
