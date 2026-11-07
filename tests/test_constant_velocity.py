import numpy as np
import pytest

from prediction.constant_velocity import ConstantVelocityPredictor


class TestConstantVelocityPredictor:
    def test_linear_motion(self) -> None:
        predictor = ConstantVelocityPredictor()
        obs = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ])
        pred = predictor.predict(obs, t_pred=3)
        assert pred.shape == (3, 2)
        np.testing.assert_allclose(pred[0], [3.0, 6.0], atol=1e-6)
        np.testing.assert_allclose(pred[1], [4.0, 8.0], atol=1e-6)
        np.testing.assert_allclose(pred[2], [5.0, 10.0], atol=1e-6)

    def test_stationary(self) -> None:
        predictor = ConstantVelocityPredictor()
        obs = np.array([[5.0, 5.0]] * 5)
        pred = predictor.predict(obs, t_pred=4)
        assert pred.shape == (4, 2)
        np.testing.assert_allclose(pred, [[5.0, 5.0]] * 4, atol=1e-6)

    def test_single_observation(self) -> None:
        predictor = ConstantVelocityPredictor()
        obs = np.array([[10.0, 20.0]])
        pred = predictor.predict(obs, t_pred=3)
        assert pred.shape == (3, 2)
        np.testing.assert_allclose(pred, [[10.0, 20.0]] * 3, atol=1e-6)

    def test_velocity_smoothing(self) -> None:
        predictor = ConstantVelocityPredictor(smooth_window=3)
        obs = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [100.0, 100.0],  # outlier
        ])
        pred = predictor.predict(obs, t_pred=1)
        # smoothed velocity should dampen the outlier effect
        assert pred[0, 0] < 120.0  # not just extrapolating from outlier

    def test_multimodal_k(self) -> None:
        predictor = ConstantVelocityPredictor()
        obs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        preds = predictor.predict(obs, t_pred=3, k=5)
        assert preds.shape == (5, 3, 2)  # K, T_pred, 2
        # For constant velocity, all K should be identical (deterministic)
        for i in range(1, 5):
            np.testing.assert_allclose(preds[i], preds[0], atol=1e-6)
