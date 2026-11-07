"""Camera Motion Compensation for SMOTP4SB Track 2 (opt-in).

Bird trajectories in this dataset are entangled with drone ego-motion: a pixel
moves because the bird flew, because the camera panned, or both. Compensation
removes the camera's contribution using its measured orientation, so the
predictor sees bird motion alone.

Built against synthetic rotations so that real IMU data plugs in by replacing
one loader -- see cmc/rotations.py.
"""
