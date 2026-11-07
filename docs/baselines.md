# Baseline results

All figures here are measured on the **validation split**: 26 of the 128 train
sequences, every fifth one, listed in `data/traj/split.json`. Higher is better
for SO-HOTA and SO-HOTA-TP, lower for distances.

## Track 1: tracking

YOLOv8-SMOT (the MVA2025 winning method) with its published weights, full-
resolution inference and its OC-SORT tracker, scored with SO-HOTA:

| detector | SO-HOTA | SO-DetA | SO-AssA | SO-DetRe | SO-DetPr |
|---|---|---|---|---|---|
| YOLOv8-SMOT-S | 0.248 | 0.142 | 0.432 | 0.162 | 0.524 |
| YOLOv8-SMOT-L | 0.310 | 0.196 | 0.490 | 0.240 | 0.507 |

Four river sequences (`0101`, `0111`, `0121`, `0126`) produce no output with
either detector. They show birds on a rocky riverbed among moving water rather
than against sky. They are 1920×1080, but the predictor adapts its input size
per sequence, and birds in the 1920×1080 sequences are larger on average
(41.7 px against 16.5 px in the 4K ones, mean √(w·h) over all training boxes),
so neither resolution nor size explains it; the boxes were checked by eye and
sit on real birds. Detection against cluttered backgrounds is an open problem
in this dataset.

## Track 2: prediction

SO-HOTA-TP on the 51 windows the scorer evaluates, with the predictor fed
ground-truth past trajectories, and fed the output of the Track 1 baseline
(detector and OC-SORT) as a submission would be. minADE₆ in pixels is given for
reference only; it does not rank submissions.

| predictor | ground truth | YOLOv8-SMOT-S | YOLOv8-SMOT-L | minADE₆, ground truth |
|---|---|---|---|---|
| `stay` (never move) | 0.127 | 0.012 | 0.013 | 370.0 |
| constant velocity | 0.158 | 0.022 | 0.023 | 726.0 |
| Kalman filter, K = 6 | 0.121 | 0.025 | 0.029 | 332.6 |
| LSTM, trained for minADE | 0.149 ± 0.016 | 0.029 ± 0.003 | 0.033 ± 0.005 | 363.1 |
| **LSTM, trained for SO-HOTA-TP** | **0.195 ± 0.006** | **0.043 ± 0.006** | **0.043 ± 0.004** | — |
| PPT (ECCV 2024), trained for minADE | 0.164 | 0.032 | 0.035 | 303.7 |
| **PPT, trained for SO-HOTA-TP** | **0.192** | **0.024** | **0.043** | 308.4 |

LSTM rows: mean ± population SD over three seeds (the minADE₆ column is seed 0).
The SO-HOTA-TP LSTM has no minADE₆: it stops tracks where it predicts the bird
leaves, so its distances cover fewer frames than the other rows'.
The validation sequences also select each learned model's checkpoint, so the
learned rows are slightly optimistic. PPT rows are one run each; with
YOLOv8-SMOT-S tracks the SO-HOTA-TP-trained PPT scores below the minADE-trained
one, which one run on 51 windows cannot separate from noise.

**Train for the metric.** SO-HOTA-TP keeps, for each bird and window, the mode
with the highest mean Dot Distance over the frames where the bird is visible,
and a predicted point counts only within a few tens of pixels of the truth
(DotD = exp(-d / 24.15) must clear thresholds from 0.05 to 0.95). It therefore
rewards precise short-term prediction and stopping a track when the bird
leaves; 41% of the birds observed in a training window disappear within the
5-second horizon. Models trained with the usual best-of-K L2 loss on
complete-future windows optimize minADE instead, and under SO-HOTA-TP they do
not reliably beat constant velocity, which is precise in the first frames and
far off at the end. Training for SO-HOTA-TP means:

- a loss shaped like the metric, 1 − exp(−d / S) averaged over visible frames,
  with the mode chosen by that same quantity (`prediction/train_sohota.py`;
  for PPT, `tools/patch_ppt_sohota.py` changes the Stage III trajectory term);
- training on windows whose future ends early, scored on their visible frames;
- for the LSTM, a visibility head that predicts when the bird leaves, so the
  track stops there (`--vis`); PPT conditions on a destination at the last
  frame and keeps complete windows only.

**The tracker costs more than any choice of predictor.** Feeding real tracks
instead of ground truth lowers SO-HOTA-TP by roughly 75 to 90% for every
predictor, more than the spread between predictors.

**Identity fragmentation is the mechanism, not box jitter.** Box centres differ
from ground truth by under 2 px, but the tracker emits 1.9 times as many ids as
the ground truth holds (474 against 248 with YOLOv8-SMOT-L; 387 with -S). A
track that breaks during the observation leaves the predictor extrapolating
from a fragment.

**Detection recall bounds everything downstream.** Tracker output matched to
the 32,999 ground-truth boxes frame by frame (Hungarian, centre distance under
S = 24.15 px):

| | output boxes | matched | recall | precision |
|---|---|---|---|---|
| YOLOv8-SMOT-S | 10,205 | 5,657 | 17.1% | 55.4% |
| YOLOv8-SMOT-L | 15,642 | 8,459 | 25.6% | 54.1% |

Improving detection and identity consistency in cluttered scenes will move the
Track 2 score further than a better motion model.

## Reproducing

```bash
# trajectory windows from the annotations
python -m prediction.build_dataset

# LSTM trained for SO-HOTA-TP, with the visibility head
python -m prediction.train_sohota --loss dotd --vis 1.0 --gpu 0 \
    --split data/traj/split.json --save runs/lstm_sohota.pt
# the same model trained for minADE, for comparison
python -m prediction.learned_models --model lstm --epochs 40 --save runs/lstm.pt

# SO-HOTA-TP on ground truth and on tracker output, any predictors
python tools/measure_tracker_gap.py --tracker-dir <tracks/> \
    --predictors stay cv kalman runs/lstm_sohota.pt <PPT Stage III checkpoint> \
    --out gap.json
```

PPT: [ppt_setup.md](ppt_setup.md).
