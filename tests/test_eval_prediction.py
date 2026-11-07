import numpy as np
import pytest

from evaluation.eval_prediction import compute_ade, compute_fde, compute_min_ade_fde, compute_miss_rate


class TestADE:
    def test_perfect_prediction(self) -> None:
        gt = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pred = gt.copy()
        assert compute_ade(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_constant_offset(self) -> None:
        gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        pred = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        assert compute_ade(pred, gt) == pytest.approx(1.0, abs=1e-6)


class TestFDE:
    def test_perfect_prediction(self) -> None:
        gt = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = gt.copy()
        assert compute_fde(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_offset_at_end(self) -> None:
        gt = np.array([[0.0, 0.0], [1.0, 0.0]])
        pred = np.array([[0.0, 0.0], [1.0, 3.0]])
        assert compute_fde(pred, gt) == pytest.approx(3.0, abs=1e-6)


class TestMinADEFDE:
    def test_best_of_k(self) -> None:
        gt = np.array([[1.0, 1.0], [2.0, 2.0]])
        preds = np.array([
            [[10.0, 10.0], [20.0, 20.0]],  # bad
            [[1.0, 1.0], [2.0, 2.0]],       # perfect
            [[5.0, 5.0], [6.0, 6.0]],       # bad
        ])  # (K=3, T=2, 2)
        min_ade, min_fde = compute_min_ade_fde(preds, gt)
        assert min_ade == pytest.approx(0.0, abs=1e-6)
        assert min_fde == pytest.approx(0.0, abs=1e-6)


class TestMissRate:
    def test_all_hit(self) -> None:
        fde_values = np.array([1.0, 2.0, 3.0])
        assert compute_miss_rate(fde_values, threshold=5.0) == pytest.approx(0.0)

    def test_all_miss(self) -> None:
        fde_values = np.array([10.0, 20.0, 30.0])
        assert compute_miss_rate(fde_values, threshold=5.0) == pytest.approx(1.0)

    def test_partial(self) -> None:
        fde_values = np.array([1.0, 10.0, 2.0, 20.0])
        assert compute_miss_rate(fde_values, threshold=5.0) == pytest.approx(0.5)
