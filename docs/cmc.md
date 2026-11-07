# IMU and camera motion compensation

Sequences filmed with the ProDrone platform carry the flight controller's
telemetry, one `imu.csv` row per frame. This repository uses it for camera
motion compensation (CMC): the camera's own motion is predicted from the IMU and
removed from the observed trajectories before prediction. It is opt-in
(`--cmc`), and sequences without `imu.csv` are left as they are.

**It is a starting point, not an improvement.** It removes about a fifth of the
camera's motion, but the baseline trained with it does not score higher
SO-HOTA-TP than the same baseline without it ([measured results](#measured-results)).

## What the IMU describes, and what it does not

The camera hangs on a gimbal with an optical zoom, and none of the gimbal
angles or the zoom ratio are logged. What reaches the image is:

| Image motion | Cause | In imu.csv |
|---|---|---|
| pan | the drone turning (the gimbal follows the heading, roughly 0.5-1 s late) | `yaw_deg`, `qw..qz`, `gyr_z` |
| pan | the pilot panning the gimbal | `gimbal_yaw_cmd` (rate input, -1..1) |
| tilt | the pilot tilting the gimbal | `gimbal_pitch_cmd` (rate input, -1..1) |
| zoom | the pilot zooming | `zoom_cmd` (+1 in, -1 out, 0 none) |
| everything at once | gimbal retracted: camera fixed to the airframe | `gimbal_retracted` |

Body pitch and roll do not reach the image while the gimbal is deployed. The
focal length changes by up to about 10x within a sequence, so a single `K` and a
rotation homography `K R K^-1` cannot describe the camera.

## The compensation

For every frame transition the image motion is modelled as a similarity about
the image centre,

```
x_{t+1} = s_t (x_t - c) + c + (dx_t, dy_t)
```

and `(dx_t, dy_t, ln s_t)` is predicted from the IMU channels around `t`
(gimbal commands at lags -8..+8 frames, the zoom command likewise, the drone's
yaw step at -45..+5, the pitch step while retracted at -3..+3). The predictor
is a ridge regression (`cmc/egomotion.py`) fitted on the training sequences
against motion measured on their own frames (`cmc/measure.py`); the lags let it
learn the gimbal's follow delay and response instead of modelling them.

Composing the per-frame similarities gives one homography per frame into the
reference frame. In `input_only` mode, the one Track 2 uses, the observed 60
frames are mapped into the last observed frame and the prediction stays in raw
pixels.

**Only the observed rows are read.** The features of a window are built from
the `imu.csv` rows of its 60 observed frames, with lags that fall outside them
set to zero, both when fitting and when predicting. This is the Track 2 rule:
future frames and future IMU must not be used.

## Using it

```bash
# fit the model (measures camera motion on the training frames; cached)
python tools/fit_egomotion.py

# how much camera motion it removes, measured on held-out frames
python tools/eval_egomotion.py --split pub_test

# the LSTM baseline with CMC, and predictions from it: the input has to be
# compensated the same way, so infer reads imu.csv from the split directory
python -m prediction.train_sohota --loss dotd --vis 1.0 --cmc \
    --split data/traj/split.json --save runs/lstm_sohota_cmc.pt
python -m prediction.infer --tracks <tracks/> --out runs/pred \
    --model runs/lstm_sohota_cmc.pt --imu-dir <SMOTP4SB_DATA>/pub_test

# the minADE-trained LSTM and PPT with CMC
python -m prediction.learned_models --model lstm --cmc --cmc-mode input_only
python -m prediction.to_ppt_format --cmc --cmc-mode input_only
python tools/score_ppt.py --ckpt <stage III checkpoint> --cmc --cmc-mode input_only
```

Settings live in `cmc/config.py` (`source`, `egomotion_model`, `imu_dir`, ...)
and can be given as JSON with `--cmc-config`. `source="rotations"` keeps the
rotation-homography path for other cameras.

## Measured results

Camera motion removed (`tools/eval_egomotion.py`; model fitted on the
training sequences, 60-frame windows compensated into their last frame,
displacement of a 5x5 image grid against the motion measured on the frames):

| split | windows | without CMC | with IMU CMC | removed |
|---|---|---|---|---|
| pub_test | 105 | 75.7 px | 59.9 px | 21.0% |

Leave-one-recording-out on the training sequences removes 21% with the shipped
settings (prediction gain 0.5), and the per-sequence worst case is 1.3x the
uncompensated displacement, against 2.2x without the gain.

Effect on trajectory prediction, in SO-HOTA-TP. The LSTM is trained as the main
baseline (`prediction.train_sohota --loss dotd --vis 1.0`), with and without
`--cmc`, with the IMU model refitted without the validation sequences
(`tools/fit_egomotion.py --exclude-val`); 26 validation sequences, 51 windows,
three seeds; `tools/measure_tracker_gap.py --imu-dir`:

| | ground-truth trajectories | YOLOv8-SMOT-L tracker |
|---|---|---|
| no CMC | 0.206 ± 0.002 | 0.046 ± 0.006 |
| IMU CMC, input_only | 0.179 ± 0.007 | 0.049 ± 0.001 |

The two rows were trained side by side; the no-CMC row is the same recipe as the
LSTM in [baselines.md](baselines.md), whose 0.195 ± 0.006 is an earlier run.
With ground-truth trajectories the compensated model is lower; with the
tracker's output the difference is within the spread between seeds. Only 5 of
the 26 validation sequences have `imu.csv` (6 of the 51 windows), too few to
judge the IMU by.

The IMU sequences were therefore also tested on their own. The 27 training
sequences with `imu.csv` come from four recordings (the `rec` column). Each
fold holds one recording out, trains on the other three and on every training
sequence without IMU, refits the IMU model on those three only, and predicts
the held-out recording from ground-truth trajectories; the four held-out parts
are scored together (41 windows, three seeds). The same LSTM and training,
with different inputs:

| LSTM input | SO-HOTA-TP |
|---|---|
| raw trajectory (the baseline) | **0.414 ± 0.008** |
| trajectory compensated with the IMU (`--cmc`) | 0.394 ± 0.008 |
| raw and compensated | 0.398 ± 0.008 |
| raw, plus gimbal and zoom commands and body yaw rate | 0.349 ± 0.005 |
| raw and compensated, plus those commands | 0.338 ± 0.002 |
| raw and compensated with motion measured on the frames (no IMU) | 0.366 ± 0.004 |

Every input that describes the camera lowers the score, and the pilot's
commands lower it most. With four recordings to learn from, they are learned
as the habits of those flights and do not carry over to another.

## Why the IMU does not help yet

On the same 27 sequences (5,196 observation windows), with the camera's
motion measured on the frames, which is more than the IMU can tell, and one
prediction per window scored by its mean DotD over the visible future frames:

| prediction | mean DotD |
|---|---|
| `stay` | 0.202 |
| constant velocity of the raw trajectory | 0.298 |
| the bird's own velocity, camera motion removed | 0.206 |
| bird and camera extrapolated separately, each with its own decay | 0.298 |
| the bird's velocity plus the **true** future camera motion | 0.423 |

- Most of what moves in the image is the camera. Two seconds after the
  observation, a world-fixed point at the bird's last position has moved 60 px
  (median); the bird has moved 66 px.
- What makes the future predictable is that the camera keeps moving, and the
  raw trajectory already carries that forward. Taking the camera's motion out
  leaves almost nothing to extrapolate, and splitting the two adds nothing.
- Knowing what the camera does next would add a lot (0.298 to 0.423), but
  that is the pilot's next move. Ridge regressions from the observed IMU, the
  observed camera motion, or both, one recording held out at a time, forecast
  it worse than holding the camera's last velocity (0.197-0.219 against 0.236,
  with the bird held still), and letting the camera's motion continue only
  while a gimbal or zoom command is active changes the score by under 0.001.

## Open directions

- Forecasting the camera's motion over the horizon: the gap between 0.298
  and 0.423 above. The IMU logs the pilot's commands but not the gimbal angles
  or the zoom ratio, and four recordings are few to learn a pilot from; the
  image may be needed as well, for instance where the bird sits in the frame,
  if the camera operator was following it.
- Track 1, where no forecast is needed: every frame's IMU is available, so the
  camera's motion between consecutive frames can be fed to the tracker's
  motion model.
- A non-linear or sequence model in place of the ridge regression; the
  zoom level is the main missing input, since a pan of the same angle moves
  the image ten times further at full zoom.
- Combining the IMU with image registration of the observed frames.
- Translation, from the NED velocities and a depth estimate.
