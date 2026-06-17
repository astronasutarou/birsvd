import numpy as np
import pytest

from birsvd.birsvd_original import birsvd_original
from birsvd.birsvd import birsvd
from birsvd.svd.parameter import BIRSVDParameter
from birsvd.svd.results import SVDResult


@pytest.mark.parametrize("func", [birsvd_original, birsvd])
@pytest.mark.parametrize("n_rank", [0, 3])
def test_birsvd_rejects_invalid_rank(func, n_rank):
    data = np.ones((3, 2))
    weight = np.ones_like(data)

    with pytest.raises(ValueError, match='"n_rank"'):
        func(data, weight, n_rank=n_rank)


@pytest.mark.parametrize("func", [birsvd_original, birsvd])
def test_birsvd_returns_svd_result(func):
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weight = np.ones_like(data)
    param = BIRSVDParameter(n_iter=2, r_type_L="TiKh", r_type_R="TiKh")

    result = func(data, weight, n_rank=1, param=param)

    assert isinstance(result, SVDResult)
    assert result.U.shape == (3, 1)
    assert result.S.shape == (1,)
    assert result.V.shape == (2, 1)
    assert result.error.shape == (2,)
    assert result.A.shape == data.shape
