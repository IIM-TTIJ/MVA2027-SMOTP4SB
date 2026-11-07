# Private test: submitting code

The final ranking comes from the private test, which is not released. Instead,
each team submits its code and trained model, and the organizers run it on the
private test sequences and score the output with the same scorer as CodaBench.
This page says what to submit so that it runs without help.

## What to submit

One archive per track (`.zip` or `.tar.gz`), or a public Git repository with a
commit hash, plus a download link for the weights, through the Google Form
linked on the challenge website before the code submission deadline. It must
contain:

- `run.sh`, the single entry point described below;
- the environment: a `Dockerfile` (recommended), or a `requirements.txt` with
  the Python and CUDA versions it was tested with;
- the trained weights, or a script that fetches them before the run;
- a short `README` naming the method and the team, matching the factsheet.

The run has **no internet access**. Everything the code needs, including
pretrained backbones, must be in the archive or fetched by the setup step.

## Hardware and time

The organizers run the code on one machine with up to four 48 GB NVIDIA GPUs
(RTX A6000 or RTX 6000 Ada), an NVIDIA driver for CUDA 12, and no internet
access. One GPU is preferred where it is enough. CPU cores and memory are not
limited beyond what such a workstation has.

There is no fixed time limit either, but the organizers run every team's code
within about a week, so the whole private test should take no more than about
a day per track. Report your public-test run time and GPUs in the submission
form. As in other academic challenges, code that needs far more than this, or
that cannot reproduce its public-test score, may not be ranked.

## Track 1: tracking

```bash
bash run.sh INPUT_DIR OUTPUT_DIR
```

`INPUT_DIR` holds one directory of frames per sequence, as in the public test,
with `imu.csv` where available, and `camera_intrinsics.json` at its root:

```
INPUT_DIR/
├── camera_intrinsics.json
└── {seq_id}/00001.jpg ...   (and imu.csv where available)
```

Write `OUTPUT_DIR/{seq_id}.txt` for every sequence, in the Track 1 submission
format. Whole sequences are given, as in the public test.

## Track 2: trajectory prediction

```bash
bash run.sh INPUT_DIR OUTPUT_DIR
```

The code receives **only the observation of each window**: for every window
that fits in a sequence, a directory with its 60 observed frames under their
original names, and the matching `imu.csv` rows where the sequence has IMU.
Future frames and future IMU rows are not in the input.

```
INPUT_DIR/
├── camera_intrinsics.json
└── {seq_id}/
    ├── 00001/                 window_start = 1
    │   ├── 00001.jpg ... 00060.jpg
    │   └── imu.csv            rows of frames 1-60 (IMU sequences only)
    ├── 00151/                 window_start = 151
    │   ├── 00151.jpg ... 00210.jpg
    │   └── imu.csv
    └── ...
```

Write `OUTPUT_DIR/prediction/{seq_id}.txt` and `OUTPUT_DIR/tracking/{seq_id}.txt`
in the Track 2 submission format, as for CodaBench:

- predictions are keyed by `window_start`, the name of the window's directory;
- `tracking/` holds your tracker's output on the observed frames of every
  window. Observations of different windows never share a frame, so track ids
  only need to be consistent within one window.

Each window is meant to be predicted from its own directory. Reading other
windows' frames to predict a window is against the rules, and the organizers
may check by running windows one at a time.

## Check before submitting

The public test can be laid out exactly as the private test will be:

```bash
python tools/private_test/package_inputs.py --track 2 --split pub_test \
    --data $SMOTP4SB_DATA --out /tmp/pub_test_t2_input
bash run.sh /tmp/pub_test_t2_input /tmp/pub_test_t2_output
python tools/make_submission.py --tracks /tmp/pub_test_t2_output/tracking \
    --pred /tmp/pub_test_t2_output/prediction --out /tmp/pub_test_t2_submission
```

and upload `track2_submission.zip` to CodaBench.

Your `run.sh` should reproduce, on this input, the public-test score you
obtained on CodaBench. The organizers do the same check on the submitted code
before running the private test, and contact the team if the two differ.
