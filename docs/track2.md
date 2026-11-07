# Track 2: trajectory prediction

The task definition: what is predicted, from what, how windows are cut, and the
submission format. The metric is defined in [so_hota_tp.md](so_hota_tp.md).

## Task

Observe **60 frames (2 s)** of UAV video and predict, for each bird, where it
will be over the next **150 frames (5 s)**, as **K = 6** candidate paths.
Video is 30 fps.

| | |
|---|---|
| observation | 60 frames |
| prediction | 150 frames |
| modes | 6 per bird |
| coordinates | box centre, pixels, in the original frame |
| ranking metric | SO-HOTA-TP |

### The input is video, not trajectories

Ground-truth tracks are distributed for the training split only. The test
splits ship frames (and IMU where available), so a submission detects and
tracks the birds itself and predicts from its own tracks. Track 2 therefore
contains Track 1, and a Track 1 tracker can be reused as it is.

Every figure measured on ground-truth trajectories is an upper bound for this
reason; [baselines.md](baselines.md) measures the gap against a real detector.

### Disappearance is part of the task

A bird that leaves the frame, is occluded or dives out of view stops having a
ground-truth position. In the training data 41% of the birds observed in a
window are not visible for all 150 frames that follow.

A submission predicts a disappearance by **stopping** a track early. Positions
emitted for frames where the bird is absent count as false positives, and frames
where it is present but nothing was emitted count as false negatives, so
predicting *when* a track ends is scored, not excluded.

## Windows

Windows are cut from the video, not from the ground truth:

- the first window starts at **frame 1** (`00001.jpg`) of each sequence;
- windows start every **150 frames**: `window_start` is 1, 151, 301, ...;
- a window spans `window_start ... window_start + 209` (60 observed, 150
  predicted) and must fit inside the sequence;
- a window is scored if at least one ground-truth bird is visible in its 60
  observed frames.

**Only a window's 60 observed frames, and their IMU rows, may be used to predict
it.** Future frames and future IMU must not be used, so camera motion over the
prediction horizon has to be estimated from the observation. The public-test
videos and IMU logs are complete so that predictions can be visualised and
analysed; in the private test, submitted code receives the observation only.

Because windows advance by 150 while each predicts 150 frames after a 60-frame
observation, consecutive windows overlap: one frame can lie in the prediction
horizon of two windows. Each window is predicted from its own observation, so
the file names the window a prediction belongs to.

Predict for every window that fits in the sequence. Predictions for windows
that are not scored are ignored.

## Submission format

A zip of two directories with one file per sequence in each:

```
submission.zip
├── prediction/{seq_id}.txt    what is scored
└── tracking/{seq_id}.txt      how your ids are matched to ground truth
```

### `prediction/{seq_id}.txt`

Six comma-separated columns, one row per predicted position:

```
window_start, frame_id, track_id, mode_k, cx, cy
```

| column | meaning |
|---|---|
| `window_start` | first observed frame of the window: 1, 151, 301, ... |
| `frame_id` | absolute frame number in the sequence, 1-indexed |
| `track_id` | your id, consistent with `tracking/{seq_id}.txt` |
| `mode_k` | 0 to 5 |
| `cx`, `cy` | predicted box centre in pixels |

A complete track-window is `150 × 6 = 900` rows, covering frames
`window_start + 60` to `window_start + 209`. Leave out the frames where you
predict the bird is out of view: a track may stop early (it leaves), start late
or resume after a gap (it was occluded). Each frame is scored on its own.

### `tracking/{seq_id}.txt`

Standard MOT columns, covering the observed frames:

```
frame_id, track_id, x, y, w, h, conf
```

`x, y` is the top-left corner in pixels. Track 1 output can be used directly.

Your track ids are your own. For each window, the scorer matches them to
ground-truth ids by Hungarian assignment on this file over the window's 60
observed frames, with the mean centre distance as cost. The predicted frames
are never used for matching, so a guess cannot be scored against whichever
ground-truth bird it happens to land near. A pair further apart than
`-S·ln(0.05)` (about 72 px with S = 24.15) is never matched, since no threshold
of the metric would accept it, and tracks sharing fewer than 5 observed frames
with a ground-truth track are not matched either. A predicted id that is absent
from the tracking file cannot be scored.

A sequence with no prediction file is scored as all misses, not skipped.

### Check before uploading

```bash
python tools/validate_submission.py submission.zip
```

It rejects duplicate rows, frames outside the horizon and predicted ids missing
from the tracking file, and warns about missing modes and modes that cover
different frames. To see a well-formed archive:

```bash
python tools/make_sample_submission.py \
    --gt SMOT4SB/annotations/train.json --sequences 0002 --out sample.zip
```

That is the `stay` baseline (every mode repeats the last observed position).

`tools/make_submission.py` builds both tracks' archives from tracking output
and predictions, and validates them.
