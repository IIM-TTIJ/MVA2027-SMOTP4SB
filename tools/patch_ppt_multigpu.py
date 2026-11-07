"""Speed up PPT training: fast batch collation, and Stage III on several GPUs. Run after patch_ppt.py.

    python tools/patch_ppt_multigpu.py
    cd PPT && CUDA_VISIBLE_DEVICES=4,5,6,7 PPT_PAST_LEN=60 \
        torchrun --standalone --nproc_per_node=4 train_PPT.py --mode ALL ... --gpu 0

If the first collective hangs until the NCCL timeout, peer-to-peer transfers
between GPUs that are not NVLink pairs are broken on that machine (seen on a
server with NVLink only within pairs); set NCCL_P2P_DISABLE=1.

Each 512-window batch is split across the GPUs. Every rank weights its loss by
its share of the batch, the gradients are summed across ranks, and only then
are they clipped and applied, so one step equals the single-GPU step on the
whole batch (dropout masks aside). Parameters start from rank 0's, all ranks
draw the same batch order each epoch, and only rank 0 validates, saves and logs.
The model is not wrapped in DistributedDataParallel: Stage III also runs its
teacher sub-modules outside the model's forward, which DDP's hooks do not
expect. With ~1M parameters the plain all-reduce costs nothing.

Without torchrun (WORLD_SIZE unset) PPT runs exactly as before.
Idempotent, and fails if an upstream pattern does not match exactly once.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PPT = ROOT / "PPT"
T = "trainer/trainer_PPT.py"

PATCHES = [
    (T,
     "            os.makedirs(self.folder_test)\n",
     "            os.makedirs(self.folder_test, exist_ok=True)   # every rank creates it at once\n",
     "run folder creation safe across ranks"),
    ("dataset_loader.py",
     "\treturn torch.Tensor(trajectories).squeeze(0), torch.Tensor(mask).squeeze(0), torch.Tensor(initial_pos).squeeze(0), torch.tensor(seq_start_end ,dtype=torch.int32).squeeze(0)",
     "\t# np.stack + from_numpy: torch.Tensor(list of arrays) copies element by element and\n"
     "\t# took most of each epoch (635 batches of 512x210x2 windows and 512x512 masks)\n"
     "\treturn (torch.from_numpy(np.stack(trajectories)).float().squeeze(0), torch.from_numpy(np.stack(mask)).float().squeeze(0),\n"
     "\t        torch.from_numpy(np.stack(initial_pos)).float().squeeze(0), torch.tensor(seq_start_end, dtype=torch.int32).squeeze(0))",
     "fast batch collation (same values, no element-wise copy)"),
    (T,
     "        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)\n",
     "        self.rank, self.world, self.share = 0, 1, 1.0\n"
     "        if int(os.environ.get('WORLD_SIZE', '1')) > 1:\n"
     "            import torch.distributed as dist\n"
     "            self.world, self.rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])\n"
     "            config.gpu = int(os.environ['LOCAL_RANK'])\n"
     "            torch.cuda.set_device(config.gpu)\n"
     "            dist.init_process_group('nccl')\n"
     "        if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)\n",
     "join the torchrun process group"),
    (T,
     "        self.start_epoch = 0\n",
     "        self.start_epoch = 0\n"
     "        if self.world > 1:\n"
     "            for p in self.model.parameters():\n"
     "                torch.distributed.broadcast(p.data, 0)\n",
     "start every rank from rank 0's parameters"),
    (T,
     "        self.logger.addHandler(stream_handler)\n",
     "        self.logger.addHandler(stream_handler)\n"
     "        if self.rank != 0:\n"
     "            self.logger.disabled = True\n",
     "log from rank 0 only"),
    (T,
     "            loss = self._train_single_epoch(epoch)\n",
     "            if self.world > 1:\n"
     "                torch.manual_seed(self.config.seed + epoch)   # same batch order on every rank\n"
     "            loss = self._train_single_epoch(epoch)\n",
     "same batch order on every rank"),
    (T,
     "            if (epoch + 1) % 1 == 0:\n",
     "            if (epoch + 1) % 1 == 0 and self.rank == 0:\n",
     "validate and save on rank 0 only"),
    (T,
     "            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)\n"
     "\n"
     "            traj_norm = trajectory - trajectory[:, self.config.past_len-1:self.config.past_len, :]\n"
     "            x = traj_norm[:, :self.config.past_len, :]\n"
     "            destination = traj_norm[:, -1:, :]\n"
     "            y = traj_norm[:, self.config.past_len:, :]\n"
     "            gt = traj_norm[:, 1:self.config.past_len+1, :]\n",
     "            trajectory, mask, initial_pos = torch.FloatTensor(trajectory).to(self.device), torch.FloatTensor(mask).to(self.device), torch.FloatTensor(initial_pos).to(self.device)\n"
     "            if self.world > 1:\n"
     "                n = trajectory.shape[0]\n"
     "                lo, hi = n * self.rank // self.world, n * (self.rank + 1) // self.world\n"
     "                trajectory, initial_pos = trajectory[lo:hi], initial_pos[lo:hi]\n"
     "                mask = mask[lo:hi, lo:hi] if mask.dim() == 2 and mask.shape[0] == n else mask[lo:hi]\n"
     "                self.share = (hi - lo) / n\n"
     "\n"
     "            traj_norm = trajectory - trajectory[:, self.config.past_len-1:self.config.past_len, :]\n"
     "            x = traj_norm[:, :self.config.past_len, :]\n"
     "            destination = traj_norm[:, -1:, :]\n"
     "            y = traj_norm[:, self.config.past_len:, :]\n"
     "            gt = traj_norm[:, 1:self.config.past_len+1, :]\n",
     "each rank takes its slice of the batch"),
    (T,
     "                loss += self.config.des_lambda_soft * self.criterionLoss(des_pred_feat, des_feat[:, -1])\n"
     "\n"
     "                loss.backward()\n",
     "                loss += self.config.des_lambda_soft * self.criterionLoss(des_pred_feat, des_feat[:, -1])\n"
     "\n"
     "                if self.world > 1:\n"
     "                    (loss * self.share).backward()      # summed over ranks = the whole-batch gradient\n"
     "                    for p in self.model.parameters():\n"
     "                        if p.grad is not None:\n"
     "                            torch.distributed.all_reduce(p.grad)\n"
     "                else:\n"
     "                    loss.backward()\n",
     "Stage III: weighted backward and gradient all-reduce"),
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
