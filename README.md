# BIRSVD

A Python implementation of the **Bi-Iterative Regularized Singular Value Decomposition (BIRSVD)** algorithm for weighted low-rank matrix approximation, based on the work of Das & Neumaier (2011).

## Installation

Install via the GitHub repository:

```bash
pip install git+https://github.com/astronasutarou/birsvd.git
```

Or download the source from the repository:

```bash
git clone https://github.com/astronasutarou/birsvd
cd birsvd
pip install .
```

### Requirements

- Python 3.9+
- numpy 2.2+
- scipy 1.16+
- scikit-learn 1.9+

## Package Contents

- `birsvd(data, weight, n_rank, param=DEFAULT_PARAM)`
  - Computes a weighted low-rank approximation of `data` using the BIRSVD algorithm using the LSQR iterative solver.
- `birsvd_original(data, weight, n_rank, param=DEFAULT_PARAM)`
  - Computes a weighted low-rank approximation of `data` using the original BIRSVD algorithm with direct linear solvers.
- `svd_imputation_with_mask(data, mask, n_rank, n_iter, ini_method='total_mean', velocity=0.1)`
  - Imputes missing values in `data` (indicated by zeros in `mask`) using iterative randomized SVD.

## Usage

```python
import numpy as np
from birsvd import birsvd_original, birsvd, imputation_with_mask
from birsvd.svd.settings import BIRSVDParameter

# Weighted low-rank approximation
m, n, r = 100, 80, 5
data   = np.random.randn(m, n)
weight = np.random.rand(m, n)   # values in (0, 1]

result = birsvd_original(data, weight, n_rank=r)
approx = result.A

# Faster variant for larger matrices
result = birsvd(data, weight, n_rank=r)
approx = result.A

# Custom parameters
param = BIRSVDParameter(n_iter=50, r_degree_L=1e-3, r_degree_R=1e-3)
result = birsvd_original(data, weight, n_rank=r, param=param)
approx = result.A

# Missing-value imputation
mask = (np.random.rand(m, n) > 0.2).astype(float)   # 20 % missing
u, s, v = imputation_with_mask(data, mask, n_rank=r, n_iter=20)
imputed = u.dot(np.diag(s)).dot(v)
```

## Background

``` bibtex
@inproceedings{Das_BIRSVD_2011,
  title={Fast Regularized Low Rank Approximation of Weighted Data Sets},
  author={Saptarshi Das and Arnold Neumaier},
  year={2011},
  url={https://api.semanticscholar.org/CorpusID:9775806}
}
```
