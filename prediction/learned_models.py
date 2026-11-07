"""Learned trajectory baselines for SMOTP4SB: LSTM / GRU / TCN / Transformer, +MDN.

All four encoders share one decoder head that emits K whole trajectories, trained
with the variety ("best-of-K") loss -- the minimum ADE over modes. That is the
direct surrogate for minADE_K, so the modes are free to spread rather than
collapsing onto the conditional mean, which is what an L2-to-the-mean loss would
force.

`--mdn` swaps the head for a mixture density output (per-mode log-sigma plus
mixture logits, trained by mixture NLL) so the model expresses calibrated
uncertainty instead of only a best guess.

Coordinates are made relative to the last observed position and divided by
POS_SCALE; raw 4K pixel values would otherwise dominate the first layer.

Usage:
    python -m prediction.learned_models --model lstm --epochs 40 --gpu 1
"""

import argparse
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn as nn

from prediction import paths
from prediction.build_dataset import gap_free_windows, load_tracks

T_OBS, T_PRED, POS_SCALE = 60, 150, 100.0


# --------------------------------------------------------------------------- #
def load_split(want_val, split, stride, cmc_cfg=None, only_with_rotations=False):
    """Windows for one split, optionally camera-motion compensated.

    Only mode="input_only" is supported here: the observation is compensated
    into the last observed frame, where the homography is the identity, so the
    origin below is still the raw anchor and the target stays raw. The metric
    is therefore computed in raw pixels, as the competition does. mode="full"
    would need the prediction mapped back to raw pixels before scoring (see
    tools/score_ppt.py) and is rejected in main().

    only_with_rotations drops sequences that have no rotation source. It is an
    explicit opt-in because it changes the evaluation set, and it applies with
    CMC off too, so a raw run can be built on exactly the same sequences as a
    compensated one -- otherwise "CMC vs none" also compares 41 sequences of
    training data against 128.
    """
    from prediction.to_ppt_format import _compensator
    val = set(json.load(open(paths.resolve(split)))["val_sequences"])
    compensate = _compensator(cmc_cfg, T_OBS)
    keep = None
    if only_with_rotations:
        from cmc.config import CMCConfig
        from cmc.pipeline import has_motion_source as has_rotation_source
        src_cfg = cmc_cfg or CMCConfig()
        keep = lambda q: has_rotation_source(q, src_cfg)
    w, skipped = [], set()
    for (q, _), (f, xy) in sorted(load_tracks().items()):
        if (q in val) != want_val:
            continue
        if keep is not None and not keep(q):
            skipped.add(q)
            continue
        for s in gap_free_windows(f, T_OBS + T_PRED, stride):
            win = xy[s:s + T_OBS + T_PRED]
            if compensate is not None:
                win = compensate(win, q, int(f[s]))
            w.append(win)
    if skipped:
        print(f"   {'val' if want_val else 'train'}: skipped {len(skipped)} sequence(s) "
              f"with no camera-motion source")
    # an empty selection (e.g. no sequence with imu.csv in this split) keeps its shape
    d = np.asarray(w, dtype=np.float32).reshape(-1, T_OBS + T_PRED, 2)
    obs, fut = d[:, :T_OBS], d[:, T_OBS:]
    origin = obs[:, -1:, :]                       # predict relative to last seen point
    return ((obs - origin) / POS_SCALE, (fut - origin) / POS_SCALE, origin)


# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, kind, d=128, layers=2, heads=4):
        super().__init__()
        self.kind = kind
        if kind in ("lstm", "gru"):
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.net = rnn(2, d, layers, batch_first=True)
        elif kind == "tcn":
            # dilations 1,2,4,8 cover the 60-frame window with 4 layers
            ch, blocks = d, []
            prev = 2
            for i in range(4):
                dil = 2 ** i
                blocks += [nn.Conv1d(prev, ch, 3, padding=dil, dilation=dil), nn.ReLU()]
                prev = ch
            self.net = nn.Sequential(*blocks)
        elif kind == "transformer":
            self.inp = nn.Linear(2, d)
            self.pos = nn.Parameter(torch.randn(1, T_OBS, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True, dropout=0.1)
            self.net = nn.TransformerEncoder(layer, layers)
        else:
            raise ValueError(kind)
        self.d = d

    def forward(self, x):                                    # x: (B, T_OBS, 2)
        if self.kind in ("lstm", "gru"):
            out, _ = self.net(x)
            return out[:, -1]
        if self.kind == "tcn":
            return self.net(x.transpose(1, 2)).mean(dim=2)
        h = self.net(self.inp(x) + self.pos)
        return h.mean(dim=1)


class Predictor(nn.Module):
    def __init__(self, kind, k, mdn, d=128):
        super().__init__()
        self.enc = Encoder(kind, d)
        self.k, self.mdn = k, mdn
        out = k * T_PRED * 2 + (k * T_PRED * 2 + k if mdn else 0)
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 512),
                                  nn.ReLU(), nn.Linear(512, out))

    def forward(self, x):
        h = self.head(self.enc(x))
        B, K = len(x), self.k
        n = K * T_PRED * 2
        mu = h[:, :n].view(B, K, T_PRED, 2)
        if not self.mdn:
            return mu, None, None
        logs = h[:, n:2 * n].view(B, K, T_PRED, 2).clamp(-6, 3)
        logit = h[:, 2 * n:]
        return mu, logs, logit


def variety_loss(mu, gt):
    """Best-of-K ADE: only the closest mode is pulled toward the target."""
    d = torch.norm(mu - gt[:, None], dim=-1).mean(dim=2)     # (B,K)
    return d.min(dim=1).values.mean()


def mdn_loss(mu, logs, logit, gt):
    """Negative log-likelihood of a diagonal Gaussian mixture over whole paths."""
    g = gt[:, None]
    ll = -0.5 * (((g - mu) / logs.exp()) ** 2) - logs - 0.5 * math.log(2 * math.pi)
    ll = ll.sum(dim=(2, 3))                                  # (B,K) joint over time & xy
    return -torch.logsumexp(torch.log_softmax(logit, dim=1) + ll, dim=1).mean()


# --------------------------------------------------------------------------- #
def minade_np(mu, gt, thr=48.0, per_window=False):
    d = np.linalg.norm(mu - gt[:, None], axis=-1)
    ade, fde = d.mean(axis=2), d[:, :, -1]
    best = ade.argmin(axis=1)
    if per_window:
        return ade.min(axis=1), fde.min(axis=1)
    return (ade.min(axis=1).mean(), fde.min(axis=1).mean(),
            (fde[np.arange(len(gt)), best] > thr).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Learned trajectory baselines")
    p.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn", "transformer"])
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--mdn", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=2, help="training window stride")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cmc", action="store_true",
                   help="compensate camera motion in the observation (input_only)")
    p.add_argument("--cmc-mode", choices=("full", "input_only"), default="input_only")
    p.add_argument("--cmc-config", default=None, help="JSON CMC config")
    p.add_argument("--save", default=None,
                   help="write the best epoch's checkpoint here, for prediction.infer")
    p.add_argument("--out", default=None,
                   help="save per-window minADE/minFDE of the best epoch to this .npz")
    p.add_argument("--only-with-rotations", action="store_true",
                   help="use only sequences that have a rotation source, with or "
                        "without --cmc, so the two can be compared on the same data")
    p.add_argument("--split", default=str(paths.SPLIT),
                   help="split.json (relative paths are taken from the repo root)")
    a = p.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    cmc_cfg = None
    if a.cmc:
        from dataclasses import replace
        from cmc.config import CMCConfig
        cmc_cfg = replace(CMCConfig.load(a.cmc_config), enabled=True).with_mode(a.cmc_mode).validate()
        if cmc_cfg.mode != "input_only":
            raise SystemExit("this baseline scores in the coordinates it predicts in; "
                             "mode=full would need decompensation (tools/score_ppt.py). "
                             "Use --cmc-mode input_only.")
        print(f"   CMC on: mode={cmc_cfg.mode}, reference={cmc_cfg.reference}")

    xt, yt, _ = load_split(False, a.split, a.stride, cmc_cfg, a.only_with_rotations)
    xv, yv, _ = load_split(True, a.split, 30, cmc_cfg, a.only_with_rotations)
    X = torch.tensor(xt).to(dev); Y = torch.tensor(yt).to(dev)
    Xv = torch.tensor(xv).to(dev); Yv_np = yv * POS_SCALE

    model = Predictor(a.model, a.k, a.mdn).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    n = len(X)
    tag = f"{a.model}{'+MDN' if a.mdn else ''}"
    print(f"{tag}: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"{n} train / {len(Xv)} val windows, seed {a.seed}, "
          f"cmc={'input_only' if cmc_cfg else 'off'}, device {dev}")

    best = (1e9, 0, 0)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            mu, logs, logit = model(X[idx])
            loss = mdn_loss(mu, logs, logit, Y[idx]) if a.mdn else variety_loss(mu, Y[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            mu, _, _ = model(Xv)
        mu_px = mu.cpu().numpy() * POS_SCALE
        ade, fde, mr = minade_np(mu_px, Yv_np)
        if ade < best[0]:
            best = (ade, fde, mr)
            best_pw = minade_np(mu_px, Yv_np, per_window=True)
            if a.save:
                # Snapshot the weights that produced the reported number, not
                # the final epoch's: the two differ by ~15 px, and the figure
                # quoted everywhere is this one.
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"   epoch {ep+1:3}  loss {tot/n:9.4f}   val minADE_{a.k} {ade:7.2f}  "
                  f"minFDE {fde:7.2f}  MR {mr:.3f}")
    print(f"  BEST {tag}: minADE_{a.k}={best[0]:.2f}  minFDE_{a.k}={best[1]:.2f}  MR@48={best[2]:.3f}")

    if a.save:
        # Everything prediction/infer.py needs to rebuild the model, so a
        # checkpoint stays usable without remembering the training flags.
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state,
            "model": a.model,
            "k": a.k,
            "mdn": a.mdn,
            "t_obs": T_OBS,
            "t_pred": T_PRED,
            "pos_scale": POS_SCALE,
            "cmc_mode": cmc_cfg.mode if cmc_cfg else None,
            "val_minADE": float(best[0]),
            "seed": a.seed,
        }, a.save)
        print(f"  saved checkpoint -> {a.save}  (best epoch, val minADE {best[0]:.2f})")

    if a.out:
        np.savez(a.out, ade_k=best_pw[0], fde_k=best_pw[1], seed=a.seed,
                 cmc="input_only" if cmc_cfg else "none",
                 only_with_rotations=a.only_with_rotations)
        print(f"  per-window errors -> {a.out}")


if __name__ == "__main__":
    main()
