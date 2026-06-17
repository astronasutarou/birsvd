import importlib

import numpy as np


def test_lsqr_forward_and_adjoint_operators_are_consistent():
    module = importlib.import_module("birsvd.birsvd_fast")
    forward = getattr(module, "__A_x")
    adjoint = getattr(module, "__Ap_x")
    rng = np.random.default_rng(0)
    m, n, n_rank = 4, 3, 2
    u = rng.normal(size=(m, n_rank))
    weight = rng.uniform(0.2, 1.0, size=(m, n))
    regularization = rng.normal(size=(n, n))
    x = rng.normal(size=n * n_rank)
    y = rng.normal(size=m * n + n * n_rank)

    lhs = np.dot(forward(x, u, weight, regularization), y)
    rhs = np.dot(x, adjoint(y, u, weight, regularization))

    np.testing.assert_allclose(lhs, rhs)


def test_lsqr_operators_match_matlab_vector_ordering():
    module = importlib.import_module("birsvd.birsvd_fast")
    forward = getattr(module, "__A_x")
    adjoint = getattr(module, "__Ap_x")
    rng = np.random.default_rng(1)
    m, n, n_rank = 4, 3, 2
    u = rng.normal(size=(m, n_rank))
    weight = rng.uniform(0.2, 1.0, size=(m, n))
    regularization = rng.normal(size=(n, n))
    x = rng.normal(size=n * n_rank)
    y = rng.normal(size=m * n + n * n_rank)

    x_re = x.reshape((n, n_rank)).T
    expected_forward_upper = weight * u.dot(x_re)
    expected_forward_lower = x_re.dot(regularization.T)
    expected_forward = np.concatenate(
        [
            expected_forward_upper.T.flatten(),
            expected_forward_lower.T.flatten(),
        ],
    )

    y_upper = y[: m * n].reshape((n, m)).T
    y_lower = y[m * n :].reshape((n, n_rank)).T
    expected_adjoint_left = u.T.dot(weight * y_upper)
    expected_adjoint_right = y_lower.dot(regularization)
    expected_adjoint = (
        expected_adjoint_left.T.flatten()
        + expected_adjoint_right.T.flatten()
    )

    np.testing.assert_allclose(
        forward(x, u, weight, regularization),
        expected_forward,
    )
    np.testing.assert_allclose(
        adjoint(y, u, weight, regularization),
        expected_adjoint,
    )
