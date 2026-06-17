import importlib

import numpy as np
import pytest

from birsvd.svd.imputation_with_mask import imputation_with_mask


def test_imputation_with_mask_returns_svd_factor_shapes():
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    mask = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    u, s, v = imputation_with_mask(data, mask, n_rank=2, n_iter=1)

    assert u.shape == (3, 2)
    assert s.shape == (2,)
    assert v.shape == (2, 3)


@pytest.mark.parametrize(
    ("ini_method", "expected"),
    [
        (
            "total_mean",
            np.array(
                [
                    [1.0, 3.25, 3.0],
                    [4.0, 5.0, 3.25],
                ]
            ),
        ),
        (
            "column_mean",
            np.array(
                [
                    [1.0, 5.0, 3.0],
                    [4.0, 5.0, 3.0],
                ]
            ),
        ),
        (
            "row_mean",
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 4.5],
                ]
            ),
        ),
        (
            "all_zeros",
            np.array(
                [
                    [1.0, 0.0, 3.0],
                    [4.0, 5.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_imputation_with_mask_initial_fill_methods(
    monkeypatch,
    ini_method,
    expected,
):
    module = importlib.import_module("birsvd.svd.imputation_with_mask")
    captured = []

    def fake_randomized_svd(matrix, n_rank):
        captured.append(matrix.copy())
        return (
            np.zeros((matrix.shape[0], n_rank)),
            np.ones(n_rank),
            np.zeros((n_rank, matrix.shape[1])),
        )

    monkeypatch.setattr(module, "randomized_svd", fake_randomized_svd)
    data = np.array(
        [
            [1.0, 10.0, 3.0],
            [4.0, 5.0, 60.0],
        ]
    )
    mask = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    imputation_with_mask(data, mask, n_rank=1, n_iter=0, ini_method=ini_method)

    np.testing.assert_allclose(captured[0], expected)


def test_imputation_with_mask_updates_only_missing_entries(monkeypatch):
    module = importlib.import_module("birsvd.svd.imputation_with_mask")
    captured = []

    def fake_randomized_svd(matrix, n_rank):
        captured.append(matrix.copy())
        return (
            np.ones((matrix.shape[0], n_rank)),
            np.array([10.0]),
            np.ones((n_rank, matrix.shape[1])),
        )

    monkeypatch.setattr(module, "randomized_svd", fake_randomized_svd)
    data = np.array([[1.0, 100.0], [3.0, 4.0]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])

    imputation_with_mask(
        data,
        mask,
        n_rank=1,
        n_iter=1,
        ini_method="all_zeros",
        velocity=0.5,
    )

    np.testing.assert_allclose(captured[1], np.array([[1.0, 5.0], [3.0, 4.0]]))


def test_imputation_with_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='Shapes of "data" and "mask"'):
        imputation_with_mask(
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            n_rank=1,
            n_iter=1,
        )


def test_imputation_with_mask_rejects_invalid_initialization_method():
    with pytest.raises(ValueError, match='Invalid "ini_method"'):
        imputation_with_mask(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            n_rank=1,
            n_iter=1,
            ini_method="invalid",
        )
