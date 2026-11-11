# MVA2027 SMOTP4SB Challenge: baseline code

Baseline code, evaluation and submission tools for the **Small Multi-Object
Tracking and Prediction for Spotting Birds (SMOTP4SB) Challenge** at MVA2027.
One repository serves both tracks.

| | Track 1: tracking | Track 2: trajectory prediction |
|---|---|---|
| input | UAV video, plus IMU where available | 2 s (60 frames) of the same |
| output | a box and an id per bird, every frame | 6 candidate paths per bird, 5 s (150 frames) ahead |
| ranking metric | SO-HOTA | SO-HOTA-TP, proposed for this challenge |
| baseline | YOLOv8-SMOT (MVA2025 winning method) | Kalman, `stay`, Seq2seq models and PPT (ECCV 2024) on Track 1's tracks |

- Challenge website: https://www.mva-org.jp/mva2027/challenge
- CodaBench, Track 1: https://www.codabench.org/competitions/18322/
- CodaBench, Track 2: https://www.codabench.org/competitions/18323/
- Dataset (Google Drive): the links will be posted here and on the challenge website on November 14, 2026

The tracks are independent competitions; you can enter either or both. Track 2
gives video, not trajectories, so a Track 2 entry tracks the birds itself
before predicting, and a Track 1 tracker can be reused as it is.

## Contents

```
YOLOv8-SMOT/   submodule: the MVA2025 winning detector and tracker
train/         Track 1 detector training (SliceTrain, mmyolo configs)
prediction/    Track 2 predictors, window builders, inference
cmc/           IMU camera motion compensation (--cmc)
evaluation/    SO-HOTA (Track 1) and SO-HOTA-TP (Track 2)
tools/         submission packaging and validation, PPT patching, measurement
scripts/       end-to-end pipeline, OpenMMLab setup
docs/          task definition, metric, results, CMC, PPT, private test
tests/
```

## Setup

```bash
git clone --recurse-submodules https://github.com/IIM-TTIJ/MVA2027-SMOTP4SB.git
cd MVA2027-SMOTP4SB

uv venv --python 3.10 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

That is enough for Track 2 prediction and for both scorers. The Track 1 detector
also needs the OpenMMLab stack:

```bash
bash scripts/setup_openmmlab.sh
```

It installs `mmcv` from OpenMMLab's prebuilt wheels at versions that work
together (mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0, torch 2.1.0 + cu121).
`pip install mmcv` and `mim install mmcv` build from source and usually fail,
and the `mmcv` 2.2.0 that `mim` resolves to by default is rejected by mmdet 3.3.0
at import time. With torch 2.1.0, keep `numpy<2` and `opencv-python<4.10`.

Requirements note: `trackeval` is installed from the `smot4sb_dev` branch of
IIM-TTIJ/TrackEval, the only branch with the SO-HOTA metric. Other branches
score with IoU without warning.

## Data

Each track has its own dataset package on Google Drive (links above), available
from November 14, 2026. The train split is the
same in both; the public-test sequences are numbered differently, so use the
package of the track you are submitting to. Download it and extract it:

```bash
python unzip.py                 # in the downloaded folder; writes ./SMOTP4SB/
ln -s /path/to/SMOTP4SB SMOT4SB # in this repository, or:
export SMOT4SB_ROOT=/path/to/SMOTP4SB
```

Every script resolves paths against the repository, so no path needs editing.
`SMOTP4SB_DATA` moves generated data (default `data/`).

The dataset README describes the layout, the annotation format, the IMU columns
and the camera intrinsics.

## Track 1: tracking

### Inference with the published weights

The YOLOv8-SMOT weights are published by their authors on Hugging Face:
[S](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_s.pth?download=true),
[M](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_m.pth?download=true),
[L](https://huggingface.co/abandoned/YOLOv8-SMOT/resolve/main/yolov8_l.pth?download=true).

```bash
cd YOLOv8-SMOT
export PYTHONPATH="$PWD:$PWD/mmyolo:$(dirname $PWD)"
python tools/predict.py --path /path/to/SMOTP4SB/pub_test \
    -c /path/to/yolov8_l.pth -f mmyolo/configs/yolov8/yolov8_l.py \
    --save_result --save_path run1 --device gpu
```

Tracks are written to `results/predictions/pub_test/run1/tracks/{seq}.txt` in
MOT format at full resolution. Pass `-f`: without it the config path may not
resolve, and the error appears as a `TypeError` about a config object. The input
size follows each sequence (3840×2160 or 1920×1080) with no setting needed.

### Training

```bash
bash train/scripts/prepare_data_coco.sh \
    /path/to/SMOTP4SB/annotations/train.json /path/to/SMOTP4SB/train   # split, slice
bash train/scripts/train.sh train/configs/yolov8_smot_l.py 0          # one GPU
bash train/scripts/train.sh train/configs/yolov8_smot_l.py 0,1,2,3    # four
bash train/scripts/eval.sh train/configs/yolov8_smot_l.py work_dirs/yolov8_smot_l/best.pth 0
```

The preparation splits train 80/20 by sequence for detector validation; that
split is separate from `data/traj/split.json`, which the Track 2 results use.
Training follows the YOLOv8-SMOT paper's SliceTrain: 1280×1280 patches with 20%
overlap for training, full frames for inference. AdamW (lr 0.001, weight decay
0.01), cosine schedule, 200 epochs with Mosaic off for the last 15, NMS IoU 0.15.

### Submission

A zip of one MOT file per sequence, `{seq_id}.txt`:

```
frame_id, track_id, x, y, w, h, conf
```

`x, y` is the top-left corner in pixels. Ids are matched to ground truth by the
scorer, so any integers work, but they must stay consistent within a sequence. A
missing sequence is scored as all misses. Build the archive with:

```bash
python tools/make_submission.py --tracks results/predictions/pub_test/run1/tracks --out runs/sub
```

### Local evaluation

```bash
python evaluation/eval_tracking.py --gt-dir <gt> --tracker-dir <tracks> --seqmap <seqmap>
```

The ground truth must be in TrackEval's layout
(`<split>/<seq>/gt/gt.txt`, `seqinfo.ini`, `seqmaps/<split>.txt`).

## Track 2: trajectory prediction

The task, the window grid and the submission format are defined in
[docs/track2.md](docs/track2.md); the metric in
[docs/so_hota_tp.md](docs/so_hota_tp.md).

### From tracks to a submission

```bash
# predictors: stay | cv | kalman | a checkpoint saved by train_sohota or learned_models
#   --save | a PPT Stage III checkpoint
python -m prediction.infer --tracks <tracks/> --out runs/pred --model runs/lstm_sohota.pt

# package both tracks' archives and validate them
python tools/make_submission.py --tracks <tracks/> --pred runs/pred --out runs/sub
python tools/validate_submission.py runs/sub/track2_submission.zip
```

The Track 2 archive holds `prediction/` and `tracking/`. Your ids mean nothing
to the scorer; the tracking file is how they are matched to ground truth, and
without it nothing can be scored.

Windows start at frame 1 and every 150 frames after. `prediction.infer` cuts
them that way by default; do not start them at the first frame your tracker
detected something.

### Training predictors

Train for the metric. SO-HOTA-TP rewards precise short-term prediction and
stopping a track when the bird leaves, which the usual best-of-K L2 loss does
not; see [docs/baselines.md](docs/baselines.md).

```bash
python -m prediction.build_dataset             # windows from annotations -> data/traj/

# LSTM trained for SO-HOTA-TP: DotD-shaped loss on windows whose future may end
# early, and a visibility head that stops the track where the bird leaves
python -m prediction.train_sohota --loss dotd --vis 1.0 --gpu 0 \
    --split data/traj/split.json --save runs/lstm_sohota.pt

# simple references and the minADE-trained LSTM
python -m prediction.eval_constant_velocity
python -m prediction.kalman
python -m prediction.learned_models --model lstm --epochs 40 --save runs/lstm.pt

# SO-HOTA-TP on ground-truth trajectories and on your tracker's output
python tools/measure_tracker_gap.py --tracker-dir <tracks/> \
    --predictors stay cv kalman runs/lstm_sohota.pt --out gap.json
```

PPT, trained for SO-HOTA-TP and on several GPUs: [docs/ppt_setup.md](docs/ppt_setup.md).

### Local evaluation

```bash
python -m evaluation.so_hota_tp --pred <prediction/seq.txt> --tracks <tracking/seq.txt> \
    --gt <gt/seq.txt> --output-json result.json
```

## End to end

From frames to both archives, optionally scored:

```bash
bash scripts/run_pipeline.sh \
    --input /path/to/SMOTP4SB/pub_test --checkpoint /path/to/yolov8_l.pth \
    --output-dir runs/pipeline [--predictor kalman] [--gt-dir <gt>]
```

The detector config is taken from the checkpoint name (`yolov8_s`, `_m`, `_l`);
pass `--config` for other names. It writes `tracking/`, `prediction/` and
`submission/track1_submission.zip`,
`submission/track2_submission.zip`. Upload each to its own competition.

## Results

On the validation split (26 train sequences); details in
[docs/baselines.md](docs/baselines.md):

- **Track 1**: SO-HOTA 0.310 with YOLOv8-SMOT-L, 0.248 with -S.
- **Track 2, ground-truth input**: SO-HOTA-TP 0.192 for PPT and 0.195 for the
  LSTM, both trained for SO-HOTA-TP; constant velocity scores 0.158 and `stay`
  0.127. Trained for minADE instead, the LSTM (0.149) does not reliably beat
  constant velocity.
- **Track 2, tracker input** (YOLOv8-SMOT-L): SO-HOTA-TP 0.043 for PPT and
  0.043 for the LSTM. Real tracks lower SO-HOTA-TP by roughly 75 to 90% for every
  predictor, more than any difference between predictors. The cause is
  detection recall (25.6% with YOLOv8-SMOT-L) and identity fragmentation.

## IMU

Camera motion compensation from the IMU is in `cmc/` (`--cmc`), as a starting
point for using the IMU. It removes about a fifth of the camera's motion from
the observation, but it does not yet improve the baseline: on recordings held
out from training, the LSTM scores 0.394 SO-HOTA-TP with it and 0.414 without,
and no other way of giving the LSTM the IMU that we tried did better. What we
tried, why it falls short, and the directions still open are in
[docs/cmc.md](docs/cmc.md).

## Private test

The final ranking runs your code on the unreleased private test. What to
submit, the `run.sh` interface of each track, the hardware and time limit,
and how to check your code on the public test laid out the same way
(`tools/private_test/package_inputs.py`): [docs/private_test.md](docs/private_test.md).

## Tests

```bash
uv pip install pytest
python -m pytest -q
```

Tests that need the dataset skip when it is not found.

## Citation

If you use the dataset, the metrics or this code, please cite the challenge
reports. The MVA2027 entry is provisional until the report is published around
May 2027.

```bibtex
@inproceedings{kondo2027smotp4sb,
  title     = {{MVA 2027 Small Multi-Object Tracking and Prediction for
                Spotting Birds Challenge: Dataset, Methods, and Results}},
  author    = {Yuki Kondo and Norimichi Ukita and Nanonn Kitbaramee and
               Masatsugu Kidode and Winners},
  booktitle = {2027 20th International Conference on Machine Vision and
               Applications (MVA)},
  year      = {2027}
}

@inproceedings{kondo2025smot4sb,
  title     = {{MVA 2025 Small Multi-Object Tracking for Spotting Birds
                Challenge: Dataset, Methods, and Results}},
  author    = {Yuki Kondo and Norimichi Ukita and Riku Kanayama and
               Yuki Yoshida and Takayuki Yamaguchi and Xiang Yu and
               Guang Liang and Xinyao Liu and Guan-Zhang Wang and
               Wei-Ta Chu and Bing-Cheng Chuang and Jia-Hua Lee and
               Pin-Tseng Kuo and I-Hsuan Chu and Yi-Shein Hsiao and
               Cheng-Han Wu and Po-Yi Wu and Jui-Chien Tsou and
               Hsuan-Chi Liu and Chun-Yi Lee and Yuan-Fu Yang and
               Kosuke Shigematsu and Asuka Shin and Ba Tran},
  booktitle = {2025 19th International Conference on Machine Vision and
               Applications (MVA)},
  year      = {2025}
}

@misc{baselinecode_mva2027_smotp4sb,
  title  = {{Baseline code for SMOTP4SB by IIM-TTIJ}},
  author = {Kondo, Yuki and Kitbaramee, Nanonn},
  url    = {https://github.com/IIM-TTIJ/MVA2027-SMOTP4SB},
  year   = {2026}
}
```

## Acknowledgements

- **YOLOv8-SMOT** by DL Team, the MVA2025 SMOT4SB Best Solution, included as a
  submodule ([paper](https://arxiv.org/abs/2507.12087),
  [code](https://github.com/Salvatore-Love/YOLOv8-SMOT)).
- **PPT** ([code](https://github.com/iSEE-Laboratory/PPT)), ECCV 2024.
- **TrackEval** and **OC-SORT**, through the IIM-TTIJ forks.
- **mmyolo** by OpenMMLab.

## Licence

Code: MIT, see [LICENSE](LICENSE). The dataset has its own licence, distributed
with it.
