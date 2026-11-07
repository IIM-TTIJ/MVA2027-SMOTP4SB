# Track 2: Short-Term Trajectory Prediction Baseline

## Overview

This module implements the trajectory prediction baseline for the MVA2027 SMOTP4SB Challenge Track 2. Participants predict future bird positions given observation-period video frames.

## Pipeline

```
Video frames (observation period)
    ↓
Track 1 baseline (YOLOv8-SMOT: detection + tracking)
    ↓
Tracked trajectories (bounding box centers over time)
    ↓
Prediction module (this component)
    ↓
Predicted future positions (T_pred frames)
```

## Baseline: Constant-Velocity Predictor

The default baseline estimates velocity from the observation window and linearly extrapolates future positions. No training is required.

- Velocity is computed from recent displacements (configurable smoothing window)
- Median-based smoothing dampens outliers from noisy detections
- Supports multimodal K predictions (all identical for this deterministic model)

## Task Specification

- **Input**: MOT-format tracking output (`frame_id, track_id, x, y, w, h, conf, class, visibility`)
- **Output**: Predicted future positions for each tracked object (`frame_id, track_id, cx, cy`)
- **Key constraint**: No ground-truth trajectories are provided; participants must perform tracking first
- **Frame rate**: Native 30fps (no subsampling)

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| T_obs | 30 frames (1.0s) | Observation window |
| T_pred | 60 frames (2.0s) | Prediction horizon |
| K (multimodal) | 1 | Number of prediction samples |
| smooth_window | 5 | Velocity smoothing window |
| stride | 30 | Sliding window stride |

### Evaluation Metrics

- **minADE_K**: Minimum Average Displacement Error over K predictions
- **minFDE_K**: Minimum Final Displacement Error over K predictions
- **Miss Rate**: Proportion of samples where minFDE exceeds threshold

## Usage

### Prediction

```bash
python -m prediction.predict \
    --tracks /path/to/tracking_output.txt \
    --output /path/to/predictions.txt \
    --t-obs 30 --t-pred 60 \
    --smooth-window 5 \
    --method constant_velocity
```

### Evaluation

```bash
python -m evaluation.eval_prediction \
    --pred /path/to/predictions.txt \
    --gt /path/to/gt.txt \
    --t-obs 30 --t-pred 60 \
    --miss-threshold 50.0 \
    --output-json results.json
```

## File Format

### Input (MOT tracking output)

```
frame_id,track_id,x,y,w,h,conf,class,visibility
1,1,100.0,200.0,20.0,15.0,1,1,1
2,1,105.0,202.0,20.0,15.0,1,1,1
```

### Output (Predicted positions)

```
frame_id,track_id,cx,cy
31,1,250.5000,310.2500
32,1,255.5000,313.2500
```

## Module Structure

| File | Description |
|------|-------------|
| `data_utils.py` | Load MOT tracks, extract sliding windows, save predictions |
| `constant_velocity.py` | Constant-velocity predictor class |
| `predict.py` | CLI entry point |

## Extending

To add a new prediction method:
1. Create a new predictor class in `prediction/` with a `predict(obs, t_pred, k)` method
2. Register the method name in `predict.py`'s `--method` choices
3. Instantiate and call in the prediction loop
