# Making PPT run on SMOTP4SB

## Summary

[PPT](https://github.com/iSEE-Laboratory/PPT) is written for pedestrian benchmarks at
**8 observed -> 12 predicted frames, K=20 modes**, in metre-scale coordinates. Track 2
needs **60 -> 150, K=6** in 4K pixels. Those numbers are hard-coded literals spread
across the model, trainer, dataloader and evaluators, so PPT will not run at our
horizon out of the box.

`python tools/patch_ppt.py` applies all 14 changes and is idempotent. After that the
full four-stage pipeline runs end to end - verified here, not just inspected.

What actually costs time if you do it by hand:

| | symptom |
|---|---|
| `block_size = 128` | our sequence is 210 tokens; **opaque CUDA indexing error**, because the upstream size guard is commented out |
| `minValue = 200` | **no checkpoint is ever saved** - the stage trains, prints a falling loss, exits 0, and writes only `train.log`, so the next stage has nothing to load |
| `d_scale = 1` | the **mode-spreading loss is dead**: `exp(-dist^2/1)` underflows to exactly 0 at pixel-scale separations, so `lambda_j` multiplies zero and diversity is never trained |
| `K = 20`, `past_len = 8`, `future_len = 12` | shape errors, or silently wrong slicing of the trajectory |

The first three are the dangerous ones: all are constants calibrated for pedestrian
data in metres, and all fail *silently* at 4K pixel scale. Everything else raises
immediately.

Three things you do **not** need to fix: the progressive-schedule masks (dead code,
never invoked), the pretrained checkpoints (none ship, and the linked pedestrian ones
cannot load into a 60->150/K=6 model), and PPT's dependency list (just install
`einops tensorboard openpyxl`).

Only **Stage III** produces a trajectory, so a comparable minADE_6 requires all four
runs - roughly 3.5 h each. For results, see [baselines.md](baselines.md).

---

## Setup

```bash
git clone https://github.com/iSEE-Laboratory/PPT.git
git -C PPT checkout 88a1c87                    # the revision the patches target
uv pip install einops tensorboard openpyxl      # missing from PPT's requirements.txt
python tools/patch_ppt.py                      # 14 patches, idempotent
python tools/patch_ppt_sohota.py               # --traj_loss dotd for Stage III
python tools/patch_ppt_multigpu.py             # fast batching; Stage III on several GPUs

python -m prediction.build_dataset --t-obs 60 --t-pred 150 --out data/traj
python -m prediction.to_ppt_format --scene sdd --out PPT/data
```

`tools/patch_ppt.py` skips already-applied patches, and **fails loudly** if an
upstream pattern no longer matches exactly once rather than half-patching.

---

## The three silent failures

**`block_size = 128`.** Sizes the position-embedding table *and* the causal-mask
buffer. Our sequence is `60 + 150 = 210` tokens, and the upstream guard is commented
out:

```python
# assert t <= self.block_size, f"Cannot forward sequence of length {t}…"
```

Too small → opaque CUDA indexing error inside the embedding lookup. Patched to 256.

**`minValue = 200` in `trainer_PPT.py` - the nastier one.** It seeds "best score so
far", and a checkpoint is only written when validation beats it. An ADE under 200 is
routine in pedestrian units; in 4K pixels our Stage II error is **307.64**, so nothing
ever beats the seed and **no model is ever saved**. The stage prints a falling loss,
looks healthy, and leaves a directory containing only `train.log` - then Stage III has
nothing to load. Patched to `float('inf')`.

**`d_scale = 1` - the quietest of the three.** The mode-spreading term is
`joint_loss = mean(exp(-dist^2 / d_scale))`, minimised to push the K endpoints apart.
Our endpoints sit ~700 px apart, and `exp(-716^2/1)` underflows to **exactly 0.0** in
float64, so `lambda_j = 100` multiplies zero: diversity was never trained in any run
that used the default. Nothing warns you. Pass `--d_scale 500000` (roughly the square
of a typical separation) to make the term active; at that scale
`exp(-716^2/5e5) = 0.36`, which is in a useful range. This is *not* in the patch
script, because the right value depends on your coordinate scale - pass it on the
command line.

---

## What else the patch changes

| File | Was | Now |
|---|---|---|
| `models/model.py` | `MLP(n_embd, 20*2, …)` | `config.num_k*2` |
| `models/model.py` | `view(…, 20, -1)` ×3, `range(20)` | `self.num_k` |
| `models/model.py` | `rand_token.repeat(1, 11, 1)` | `config.future_len - 1` |
| `trainer/evaluations.py` | `traj_norm[:, 8:, :]`, `repeat(1, 20, …)` | `config.past_len`, `config.num_k` |
| `trainer/test_final_trajectory.py` | `"future_len": 12` | `config.future_len` |
| `models/model_test_trajectory.py` | `20` literals | `self.num_k` |
| `dataset_loader.py` | `b[:, 7, :]` | `b[:, PAST_LEN-1, :]` |
| `train_PPT.py` | `past_len=8, future_len=12` | `60`, `150`, `--num_k 6` |

**Data format gotchas** - both handled by `to_ppt_format.py`:

- Write pickles as scene **`sdd`** into `PPT/data/`; PPT's existing `sdd` branch
  expects exactly that naming, so no extra code change is needed.
- **Batches must be uniform** - PPT stacks them with `np.array()`, so a trailing
  partial batch raises `inhomogeneous shape`. The converter drops the remainder and
  reports how many.

**If you hand-write patches:** in `trainer/evaluations.py` the `future_rep = …` line
appears twice, once commented out - match on leading indentation or you patch both.

---

## Running it

The four stages, in the configuration that produced the numbers in
[baselines.md](baselines.md):

```bash
cd PPT
COMMON="--dataset_name sdd --past_len 60 --future_len 150 --num_k 6 \
        --block_size 256 --n_layer 3 --gpu 0 --max_epochs 80 --d_scale 500000"

# I: short-term dynamics
PPT_PAST_LEN=60 python train_PPT.py --info smot_ST --mode Short_term $COMMON \
    --learning_rate 0.001

# II-a: destination warm-up. Sweep the spreading weight -- 0 / 10 / 100 were
# tried and lambda_j=10 won (destination ADE 252.88 against 289.41 and 255.31).
PPT_PAST_LEN=60 python train_PPT.py --info smot_DW --mode Des_warm $COMMON \
    --model_Pretrain <ST_ckpt> --learning_rate 0.001 --lambda_j 10

# II-b: destination fine-tune
PPT_PAST_LEN=60 python train_PPT.py --info smot_LT --mode Long_term $COMMON \
    --model_Pretrain <DW_ckpt> --learning_rate 0.0001 --lambda_j 10

# III: full trajectory, distilling from I and II, trained for SO-HOTA-TP
PPT_PAST_LEN=60 python train_PPT.py --info smot_ALL --mode ALL $COMMON \
    --model_Pretrain <LT_ckpt> --model_LT <LT_ckpt> --model_ST <ST_ckpt> \
    --learning_rate 0.0015 --lambda_j 10 --traj_lambda_soft 5 --des_lambda_soft 0.5 \
    --traj_loss dotd
```

**`--traj_loss dotd` trains Stage III for the ranking metric.** Upstream, the
trajectory term is mean L2, which is minADE. `tools/patch_ppt_sohota.py` replaces
it with S (1 - exp(-d / S)) + 0.012 d (d in pixels, S = 24.15): the slope of L2
for small errors, bounded by S for large ones, so its weight against the
destination and distillation terms is unchanged. The checkpoint is chosen by the
validation mean best-mode DotD instead of 3 ADE + FDE. Stages I and II are left as
they are, and Stage III still trains on complete windows, because it is
conditioned on the destination at the last frame. Results:
[baselines.md](baselines.md).

**Stage III on several GPUs.** After `tools/patch_ppt_multigpu.py`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PPT_PAST_LEN=60 \
    python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_PPT.py --mode ALL ... --gpu 0
```

Each 512-window batch is split across the GPUs and the gradients are combined so
that a step equals the single-GPU step. If the first collective hangs until the
NCCL timeout, peer-to-peer transfers between GPUs that are not NVLink pairs are
broken on that machine; set `NCCL_P2P_DISABLE=1`. The same patch builds batches
with `np.stack` instead of `torch.Tensor(list)`, which had taken most of each
epoch.

Score any Stage III checkpoint with SO-HOTA-TP, on ground-truth trajectories and
on tracker output, by passing its path as a predictor:

```bash
python tools/measure_tracker_gap.py --tracker-dir <tracks/> \
    --predictors PPT/training/training_ALL/<run>/model_<date> --out gap.json
```

**`--d_scale 500000` is not optional.** It scales the spreading term into a range
where it is numerically active; leave it out and `lambda_j` changes nothing, so the
Stage II-a sweep silently compares three identical runs.

`PPT_PAST_LEN` is read by the dataloader's module-level helper - set it or the
relative-positioning anchor stays at frame 7.

**Run every command from inside `PPT/`.** It resolves `./data/...` and `training/...`
against the working directory, so invoking `python PPT/train_PPT.py` from the repo root
fails with `FileNotFoundError: ./data/social_sdd_train_512_0_100.pickle` - it is looking
at the *project's* `data/`, not PPT's. Checkpoint paths passed via `--model_Pretrain`
must likewise be PPT-relative.

**Epochs scale with the data.** At stride 1 the training set is 65,165
windows = 127 batches, which the loader replays five times per epoch (635 batches),
five times denser than stride=5. Use **80 epochs**, not 400, to keep total gradient
steps comparable - otherwise a stage takes ~18 h instead of ~3.5 h. Measured on one
RTX A6000 before the batching fix: 3.5 to 6 h per stage, about 20 h for all four.

**Only Stage III produces a trajectory.** Stage I predicts one next position; Stage II
outputs `(B, K, 2)` - six endpoints, no path. A minADE_6 exists only after Stage III,
so budget all four runs (~3.5 h each).

| stage | verified | result |
|---|---|---|
| I | 400 epochs | loss 40,675 → 42.5 |
| II-a | runs, loads Stage I | destination error 307.64 px |
| II-b | same path as II-a | - |
| III | runs, distils from I + II | minADE_6 261.55 after 1 epoch |

---

## Two things you do *not* need to fix

**The progressive-schedule masks.** `pred_3points` / `pred_11points` /
`pred_20points` look like the paper's curriculum and look impossible to generalise to
150 steps. They are **never invoked** - every call site passes `'causal'` or `'all'`:

```
model.py:377 'all'   model.py:394 'causal'   model.py:425 'all'
model_test_trajectory.py:60 'all'   trainer_PPT.py:326 'causal'
```

**Pretrained weights.** None ship with the repo; the README's Google Drive models are
pedestrian, `8/12/K=20`, `block_size 128`. They cannot load into a 60→150/K=6 model -
`predictor_Des`, `decoder_x/y`, `traj_rand_fut_token` (11 vs 149 prompts) and `wpe`
(128 vs 210 rows) all mismatch. A warm start of the transformer blocks is possible
(`load_state_dict(..., strict=False)` over shape-compatible tensors) but speeds
convergence at best; it does not remove the need to train.
