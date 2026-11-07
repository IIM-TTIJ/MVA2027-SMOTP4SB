"""Train the learned baselines with disappearance-aware (masked) supervision.

Windows may have truncated futures (the bird leaves frame), so both the loss and
the metric are averaged over GT-present frames only -- which is how minADE_6 is
defined here. Training on complete-future windows alone, as the original recipe implies,
throws away 73% of the data and never teaches the model the majority case.

Usage:
    python -m prediction.train_masked --model lstm --epochs 40 --gpu 0
"""
import argparse

import torch
import torch.nn as nn

from prediction import paths
from prediction.build_dataset_masked import build, masked_metrics
from prediction.learned_models import POS_SCALE, Predictor

T_OBS, T_PRED, K = 60, 150, 6


def masked_variety_loss(mu, y, m):
    """Best-of-K, averaged over valid frames only. mu:(B,K,T,2) y:(B,T,2) m:(B,T)"""
    d = torch.norm(mu - y[:, None], dim=-1)                 # (B,K,T)
    mm = m[:, None, :].float()
    ade = (d * mm).sum(2) / mm.sum(2).clamp(min=1)
    return ade.min(1).values.mean()


def main():
    p = argparse.ArgumentParser(description="Masked (disappearance-aware) training")
    p.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn", "transformer"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    to, tf, tm = build(T_OBS, T_PRED, a.stride, a.split, False)
    vo, vf, vm = build(T_OBS, T_PRED, 30, a.split, True)
    org = to[:, -1:, :]
    X = torch.tensor((to - org) / POS_SCALE).to(dev)
    Y = torch.tensor((tf - org) / POS_SCALE).to(dev)
    M = torch.tensor(tm).to(dev)
    vorg = vo[:, -1:, :]
    XV = torch.tensor((vo - vorg) / POS_SCALE).to(dev)
    full = vm.all(1)
    print(f"{a.model}: train {len(X)} windows ({(~tm.all(1)).mean()*100:.0f}% disappearing), "
          f"val {len(XV)} ({(~full).mean()*100:.0f}% disappearing)")

    m = Predictor(a.model, K, False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best = (1e9, None)
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = masked_variety_loss(m(X[idx])[0], Y[idx], M[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        sch.step(); m.eval()
        with torch.no_grad():
            mu = m(XV)[0].cpu().numpy() * POS_SCALE + vorg[:, None]
        r = masked_metrics(mu, vf, vm)
        if r["minADE"] < best[0]:
            best = (r["minADE"], mu.copy())
    mu = best[1]
    for tag, sel in (("all", slice(None)), ("complete", full), ("disappearing", ~full)):
        r = masked_metrics(mu[sel], vf[sel], vm[sel])
        lbl = "D-ADE / D-FDE" if tag == "disappearing" else "minADE / minFDE"
        print(f"  {tag:<14} n={r['n']:<4} {lbl} = {r['minADE']:7.2f} / {r['minFDE']:7.2f}   MR@48={r['MR']:.3f}")


if __name__ == "__main__":
    main()
