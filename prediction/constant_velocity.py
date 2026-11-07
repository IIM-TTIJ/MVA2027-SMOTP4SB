"""Constant-velocity trajectory predictor.

Estimates velocity from the observation window and linearly extrapolates.
Serves as a simple, training-free baseline for Track 2.
"""

import numpy as np


class ConstantVelocityPredictor:
    """Predict future positions by linear extrapolation of recent velocity.

    Args:
        smooth_window: Number of recent velocity estimates to average.
            1 = use only the last displacement. Higher values smooth noise.
    """

    def __init__(self, smooth_window: int = 1) -> None:
        self.smooth_window = max(1, smooth_window)

    def predict(
        self,
        obs: np.ndarray,
        t_pred: int,
        k: int = 1,
    ) -> np.ndarray:
        """Predict future trajectory from observation.

        Args:
            obs: (T_obs, 2) array of observed (cx, cy) positions.
            t_pred: Number of future frames to predict.
            k: Number of prediction samples. For constant velocity,
               all samples are identical (deterministic model).

        Returns:
            If k == 1: (T_pred, 2) array of predicted positions.
            If k > 1: (K, T_pred, 2) array.
        """
        if len(obs) < 2:
            last_pos = obs[-1]
            single = np.tile(last_pos, (t_pred, 1))
            return np.stack([single] * k) if k > 1 else single

        velocities = np.diff(obs, axis=0)  # (T_obs - 1, 2)
        n_avg = min(self.smooth_window, len(velocities))
        # Median (rather than mean) of the windowed velocities is used so a
        # single outlier displacement (e.g. a detector glitch) does not
        # dominate the extrapolated velocity.
        smoothed_vel = np.median(velocities[-n_avg:], axis=0)

        last_pos = obs[-1]
        predicted = np.array([
            last_pos + smoothed_vel * (t + 1) for t in range(t_pred)
        ])

        if k > 1:
            return np.stack([predicted] * k)
        return predicted
