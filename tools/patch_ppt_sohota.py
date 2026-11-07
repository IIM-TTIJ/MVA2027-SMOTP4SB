"""Let PPT's Stage III train for SO-HOTA-TP (--traj_loss dotd). Run after patch_ppt.py.

    python tools/patch_ppt.py
    python tools/patch_ppt_sohota.py

Stage III fits its trajectory with mean L2, which is minADE. SO-HOTA-TP keeps,
per track and window, the mode with the highest mean DotD = exp(-d / S) over
the visible frames and gives nothing for points much farther than S. With
--traj_loss dotd the trajectory term becomes

    S * (1 - exp(-d / S)) + dotd_lam * d            (d in pixels, S = 24.15)

which has the slope of L2 for small errors and is bounded by S for large ones,
so the balance against the destination and distillation terms is kept. The
checkpoint is then chosen by validation mean best-mode DotD instead of
3 * ADE + FDE. --traj_loss l2 (the default) leaves PPT as it was.

PPT conditions Stage III on the ground-truth destination at the last predicted
frame, so windows whose bird disappears before then cannot be used as they are;
it trains on complete windows, unlike prediction/train_sohota.py.

Idempotent, and fails if an upstream pattern does not match exactly once.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PPT = ROOT / "PPT"

PATCHES = [
    ("train_PPT.py",
     '    parser.add_argument("--lambda_desloss", type=float, default=1)\n',
     '    parser.add_argument("--lambda_desloss", type=float, default=1)\n'
     '    parser.add_argument("--traj_loss", choices=("l2", "dotd"), default="l2",\n'
     '                        help="Stage III trajectory loss; dotd trains for SO-HOTA-TP")\n'
     '    parser.add_argument("--dotd_s", type=float, default=24.15, help="DotD scale S in pixels")\n'
     '    parser.add_argument("--dotd_lam", type=float, default=0.012, help="weight of plain L2 with dotd")\n',
     "options for the SO-HOTA-TP trajectory loss"),
    ("trainer/trainer_PPT.py",
     "                loss += self.L2_Loss(pred_results, y)\n",
     "                if self.config.traj_loss == 'dotd':\n"
     "                    d = torch.norm(pred_results - y, dim=-1)\n"
     "                    s = self.config.dotd_s\n"
     "                    loss += (s * (1 - torch.exp(-d / s))).mean() + self.config.dotd_lam * d.mean()\n"
     "                else:\n"
     "                    loss += self.L2_Loss(pred_results, y)\n",
     "Stage III trajectory term: DotD-shaped instead of L2"),
    ("trainer/trainer_PPT.py",
     "                if self.config.mode == 'ALL':\n"
     "                    if 3 * currentValue + fde_ < minValue:\n",
     "                if self.config.mode == 'ALL' and self.config.traj_loss == 'dotd':\n"
     "                    dotd = evaluate_trajectory_dotd(self.val_loader, self.model, self.config, self.device)\n"
     "                    self.logger.info('val best-mode DotD: {:.4f}'.format(dotd))\n"
     "                    if -dotd < minValue:\n"
     "                        minValue = -dotd\n"
     "                        self.logger.info('max DotD value: {}'.format(dotd))\n"
     "                        torch.save(self.model, self.folder_test + 'model_' + self.name_test)\n"
     "                elif self.config.mode == 'ALL':\n"
     "                    if 3 * currentValue + fde_ < minValue:\n",
     "Stage III checkpoint chosen by validation best-mode DotD"),
    ("trainer/evaluations.py",
     "def evaluate_trajectory(dataset, model, config, device):\n",
     "def evaluate_trajectory_dotd(dataset, model, config, device):\n"
     "    \"\"\"Mean over windows of the best mode's mean DotD over the predicted frames.\"\"\"\n"
     "    total, samples = 0.0, 0\n"
     "    with torch.no_grad():\n"
     "        for (trajectory, mask, initial_pos, seq_start_end) in dataset:\n"
     "            trajectory = torch.FloatTensor(trajectory).to(device)\n"
     "            traj_norm = trajectory - trajectory[:, config.past_len-1:config.past_len, :]\n"
     "            x = traj_norm[:, :config.past_len, :]\n"
     "            output = model.get_trajectory(x, trajectory[:, :config.past_len, :], seq_start_end,\n"
     "                                          trajectory[:, config.past_len-1, :]).data\n"
     "            fut = traj_norm[:, config.past_len:, :].unsqueeze(1)\n"
     "            dotd = torch.exp(-torch.norm(output - fut, dim=3) / config.dotd_s).mean(dim=2)\n"
     "            total += dotd.max(dim=1).values.sum().item()\n"
     "            samples += dotd.shape[0]\n"
     "    return total / samples\n"
     "\n\n"
     "def evaluate_trajectory(dataset, model, config, device):\n",
     "validation metric: mean best-mode DotD"),
    ("trainer/trainer_PPT.py",
     "from trainer.evaluations import *\n",
     "from trainer.evaluations import *\n"
     "from trainer.evaluations import evaluate_trajectory_dotd\n",
     "import the DotD validation metric"),
]


def main() -> int:
    if not PPT.is_dir():
        print(f"no PPT checkout at {PPT}; clone it and run tools/patch_ppt.py first")
        return 1
    applied = skipped = 0
    for rel, old, new, what in PATCHES:
        p = PPT / rel
        t = p.read_text(encoding="utf-8")
        if new in t:
            skipped += 1
            continue
        n = t.count(old)
        if n != 1:
            print(f"FAIL {rel}: expected the pattern for '{what}' once, found {n}")
            return 1
        p.write_text(t.replace(old, new), encoding="utf-8")
        applied += 1
        print(f"  {rel}: {what}")
    print(f"applied {applied}, already present {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
