import numpy as np
import pytest

from birsvd.svd.functions import __get_regularization_matrix, __legendre_polys


def test_legendre_polys_returns_expected_values():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    actual = __legendre_polys(n_rank=5, x=x)

    expected = np.column_stack(
        [
            np.ones_like(x),
            x,
            (3 * x**2 - 1) / 2,
            (5 * x**3 - 3 * x) / 2,
            (35 * x**4 - 30 * x**2 + 3) / 8,
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_legendre_polys_rejects_non_vector_input():
    with pytest.raises(ValueError, match='"x" should be a vector'):
        __legendre_polys(3, np.array([[0.0, 1.0]]))


@pytest.mark.parametrize(
    ("r_type", "expected"),
    [
        ("TiKh", np.eye(5)),
        (
            "2ndOrderDiff_acc2",
            np.array(
                [
                    [-1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.5, -1.0, 0.5, 0.0, 0.0],
                    [0.0, 0.5, -1.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5, -1.0, 0.5],
                    [0.0, 0.0, 0.0, 1.0, -1.0],
                ]
            ),
        ),
    ],
)
def test_get_regularization_matrix_exact_small_matrices(r_type, expected):
    actual = __get_regularization_matrix(5, r_type)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r_type", "stencil"),
    [
        ("2ndOrderDiff_acc4", [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        (
            "2ndOrderDiff_acc6",
            [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
        ),
        (
            "2ndOrderDiff_acc8",
            [
                -1 / 560,
                8 / 315,
                -1 / 5,
                8 / 5,
                -205 / 72,
                8 / 5,
                -1 / 5,
                8 / 315,
                -1 / 560,
            ],
        ),
    ],
)
def test_get_regularization_matrix_interior_stencils(r_type, stencil):
    n = 20
    center = 10
    actual = __get_regularization_matrix(n, r_type)

    expected_row = np.zeros(n)
    offset = len(stencil) // 2
    expected_row[center - offset : center + offset + 1] = stencil

    assert actual.shape == (n, n)
    np.testing.assert_allclose(actual[center], expected_row)


def test_get_regularization_matrix_rejects_unknown_type():
    with pytest.raises(ValueError, match='Wrong "r_type"'):
        __get_regularization_matrix(5, "unknown")
