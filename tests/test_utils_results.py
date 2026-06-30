import numpy as np
import pytest

from birsvd.utils.results import SVDResult


def test_svd_result_stores_values():
    u = np.array([[1.0], [2.0], [3.0]])
    s = np.array([2.0])
    v = np.array([[4.0], [5.0]])
    error = np.array([0.3, 0.1])

    result = SVDResult(U=u, S=s, V=v, error=error)

    assert result.U is u
    assert result.S is s
    assert result.V is v
    assert result.error is error
    np.testing.assert_allclose(result.A, np.array([[8.0, 10.0],
                                                   [16.0, 20.0],
                                                   [24.0, 30.0]]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"U": np.ones(2)}, '"U" should be a matrix'),
        ({"S": np.ones((2, 1))}, '"S" should be a vector'),
        ({"V": np.ones(2)}, '"V" should be a matrix'),
        ({"U": np.ones((2, 3))}, 'columns in "U"'),
        ({"V": np.ones((2, 3))}, 'columns in "V"'),
        ({"error": np.ones((2, 1))}, '"error" should be a vector'),
    ],
)
def test_svd_result_rejects_inconsistent_shapes(kwargs, message):
    values = {
        "U": np.eye(2),
        "S": np.array([2.0, 1.0]),
        "V": np.eye(2),
        "error": np.array([]),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SVDResult(**values)
