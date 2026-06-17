import numpy as np
import pytest

from birsvd.birsvd import birsvd
from birsvd.birsvd_fast import birsvd_fast


@pytest.mark.parametrize("func", [birsvd, birsvd_fast])
@pytest.mark.parametrize("n_rank", [0, 3])
def test_birsvd_rejects_invalid_rank(func, n_rank):
    data = np.ones((3, 2))
    weight = np.ones_like(data)

    with pytest.raises(ValueError, match='"n_rank"'):
        func(data, weight, n_rank=n_rank)
