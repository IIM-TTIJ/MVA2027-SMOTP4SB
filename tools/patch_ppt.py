"""Patch a fresh PPT clone for SMOTP4SB (60 observed -> 150 predicted, K=6).

PPT is written for pedestrian benchmarks with past_len=8, future_len=12, K=20,
and those numbers appear as literals in the model, the trainer and the
evaluators. This script rewrites them into config lookups so the same code can
run at our horizon, and is idempotent -- re-running it is a no-op.

Run after cloning PPT (from any directory):

    git clone https://github.com/iSEE-Laboratory/PPT.git
    python tools/patch_ppt.py

See docs/ppt_setup.md for what each change does and why.
"""

import sys
from pathlib import Path

# Anchored to the repo, not the working directory, so the script can be run
# from anywhere rather than only from the repo root.
ROOT = Path(__file__).resolve().parent.parent
PPT = ROOT / "PPT"

# (file, old, new, description). Every `old` must appear exactly once.
PATCHES = [
    # ---- K: number of predicted modes -------------------------------------
    ("models/model.py",
     "self.predictor_Des = MLP(config.n_embd, 20*2, (512, 512, 512))",
     "self.predictor_Des = MLP(config.n_embd, config.num_k*2, (512, 512, 512))",
     "destination head emits K endpoints, not 20"),

    ("models/model.py",
     "        pred_des = self.predictor_Des(feat[:, -1])\n"
     "        destination_prediction = pred_des.view(pred_des.size(0), 20, -1)\n"
     "\n"
     "        # generate N=20 future trajectories\n"
     "        for i in range(20):",
     "        pred_des = self.predictor_Des(feat[:, -1])\n"
     "        destination_prediction = pred_des.view(pred_des.size(0), self.num_k, -1)\n"
     "\n"
     "        # generate N=num_k future trajectories\n"
     "        for i in range(self.num_k):",
     "generation loop iterates over K modes"),

    ("models/model.py",
     "            pred_des = self.predictor_Des(feat[:, -1])    # generate 20 destinations for each trajectory\n"
     "            return pred_des.view(pred_des.size(0), 20, -1)",
     "            pred_des = self.predictor_Des(feat[:, -1])    # generate K destinations for each trajectory\n"
     "            return pred_des.view(pred_des.size(0), self.num_k, -1)",
     "Des_warm / Long_term reshape uses K"),

    ("models/model.py",
     "            pred_des = pred_des.view(pred_des.size(0), 20, -1)\n"
     "            distances = torch.norm(destination - pred_des, dim=2)",
     "            pred_des = pred_des.view(pred_des.size(0), self.num_k, -1)\n"
     "            distances = torch.norm(destination - pred_des, dim=2)",
     "ALL-mode reshape uses K"),

    # ---- future prompt tokens: future_len - 1, the last step is the endpoint
    ("models/model.py",
     "self.traj_rand_fut_token = nn.Parameter(deepcopy(pretrained_model.rand_token.data.repeat(1, 11, 1)))",
     "self.traj_rand_fut_token = nn.Parameter(deepcopy(pretrained_model.rand_token.data.repeat(1, config.future_len - 1, 1)))",
     "prompt count follows future_len (destination supplies the final step)"),

    # ---- evaluators: past_len and K were literals ---------------------------
    # Leading newline+indent matters: the same text also appears commented out
    # a few lines below, and a bare substring would match both.
    ("trainer/evaluations.py",
     "\n            future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)",
     "\n            future_rep = traj_norm[:, config.past_len:, :].unsqueeze(1).repeat(1, config.num_k, 1, 1)",
     "evaluation splits at past_len and scores K modes"),

    ("trainer/test_final_trajectory.py",
     '            "future_len": 12,',
     '            "future_len": config.future_len,',
     "test config no longer pins future_len to 12"),

    ("trainer/test_final_trajectory.py",
     "future_rep = traj_norm[:, 8:, :].unsqueeze(1).repeat(1, 20, 1, 1)",
     "future_rep = traj_norm[:, self.config.past_len:, :].unsqueeze(1).repeat(1, self.config.num_k, 1, 1)",
     "final-trajectory test splits at past_len and scores K modes"),

    ("models/model_test_trajectory.py",
     "        destination_prediction = pred_des.view(pred_des.size(0), 20, -1)",
     "        destination_prediction = pred_des.view(pred_des.size(0), self.num_k, -1)",
     "test-time reshape uses K"),

    ("models/model_test_trajectory.py",
     "        for i in range(20):",
     "        for i in range(self.num_k):",
     "test-time generation loop uses K"),

    # ---- expose num_k on the model, and set our horizon as the default -----
    ("models/model.py",
     "        self.past_len = config.past_len\n"
     "        self.future_len = config.future_len\n"
     "        self.mode = config.mode\n",
     "        self.past_len = config.past_len\n"
     "        self.future_len = config.future_len\n"
     "        self.num_k = config.num_k\n"
     "        self.mode = config.mode\n",
     "num_k attribute on the main model"),

    ("train_PPT.py",
     '    parser.add_argument("--past_len", type=int, default=8, help="length of past (in timesteps)")\n'
     '    parser.add_argument("--future_len", type=int, default=12, help="length of future (in timesteps)")',
     '    parser.add_argument("--past_len", type=int, default=60, help="length of past (in timesteps)")\n'
     '    parser.add_argument("--future_len", type=int, default=150, help="length of future (in timesteps)")\n'
     '    parser.add_argument("--num_k", type=int, default=6, help="number of predicted modes (K)")',
     "SMOTP4SB horizon and K as defaults"),

    # block_size caps both the position table and the causal-mask buffer, and
    # the guard on it is commented out upstream, so too small fails obscurely.
    ("train_PPT.py",
     '    parser.add_argument("--block_size", type=int, default=128)',
     '    parser.add_argument("--block_size", type=int, default=256,\n'
     '                        help="must exceed past_len + future_len (210 for 60+150)")',
     "block_size large enough for a 210-step sequence"),

    # ---- checkpointing threshold assumes pedestrian-scale coordinates -------
    # minValue seeds the "best so far" score. In SDD/ETH units an ADE under 200
    # is ordinary, but SMOT4SB works in 4K pixels where stage II/III errors are
    # in the hundreds, so nothing ever beats the seed and no model is ever
    # saved. Start from infinity so the first epoch always checkpoints.
    ("trainer/trainer_PPT.py",
     "        minValue = 200\n",
     "        minValue = float('inf')\n",
     "checkpoint threshold not pinned to pedestrian-scale ADE"),

    # ---- dataloader: index 7 is 'last observed frame' ----------------------
    ("dataset_loader.py",
     "starting_pos = b[:, 7, :].copy()/1000 #starting pos is end of past, start of future. scaled down.",
     "starting_pos = b[:, PAST_LEN - 1, :].copy()/1000 #starting pos is end of past, start of future. scaled down.",
     "relative-positioning anchor follows past_len"),

    ("dataset_loader.py",
     "from models.preprocessing import *",
     "from models.preprocessing import *\n"
     "import os\n"
     "# Set by train_PPT.py before the dataset is constructed.\n"
     "PAST_LEN = int(os.environ.get('PPT_PAST_LEN', 8))",
     "expose past_len to the module-level helper", "PPT_PAST_LEN"),
]

def apply(path: Path, old: str, new: str, desc: str, marker: str | None = None) -> str:
    """Replace `old` with `new`, exactly once.

    `marker` is text unique to the patched result, used to detect an
    already-applied patch. It must be supplied whenever `new` contains `old`
    (an additive patch), because then `old` still matches afterwards and the
    naive check would re-apply the patch on every run.
    """
    text = path.read_text()
    if (marker or new) in text and (marker is not None or old not in text):
        return f"  skip (already applied): {desc}"
    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"FAILED in {path}: expected exactly 1 occurrence of:\n  {old[:90]!r}\n"
            f"found {n}. The upstream file has changed -- re-check the patch."
        )
    path.write_text(text.replace(old, new))
    return f"  ok: {desc}"


def main() -> None:
    if not PPT.is_dir():
        sys.exit("PPT/ not found. git clone https://github.com/iSEE-Laboratory/PPT.git first.")
    print("Patching PPT for SMOTP4SB (past=60, future=150, K=6)\n")
    for entry in PATCHES:
        rel, old, new, desc = entry[:4]
        marker = entry[4] if len(entry) > 4 else None
        print(apply(PPT / rel, old, new, desc, marker))
    # model_test_trajectory.py also needs num_k on the module
    p = PPT / "models/model_test_trajectory.py"
    t = p.read_text()
    if "self.num_k" in t and "self.num_k = config.num_k" not in t:
        t = t.replace("        self.future_len = config.future_len\n",
                      "        self.future_len = config.future_len\n        self.num_k = config.num_k\n", 1)
        p.write_text(t)
        print("  ok: num_k attribute on test model")
    print("\nDone. The progressive-schedule masks in models/model.py")
    print("('pred_3points' / 'pred_11points') are never invoked and need no change;")
    print("pass --d_scale on the command line (docs/ppt_setup.md).")


if __name__ == "__main__":
    main()
