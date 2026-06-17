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

### `birsvd(data, weight, n_rank, param=DEFAULT_PARAM)`

Computes a weighted low-rank approximation of `data` using the BIRSVD algorithm.
Solves the normal equations directly via Cholesky / LU decompositions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `ndarray (m, n)` | Input data matrix |
| `weight` | `ndarray (m, n)` | Non-negative weight matrix (same shape as `data`) |
| `n_rank` | `int` | Target rank of the approximation |
| `param` | `BIRSVDParameter` | Algorithm settings (optional) |

Returns the low-rank approximation matrix of shape `(m, n)`.

### `birsvd_fast(data, weight, n_rank, param=DEFAULT_PARAM)`

A faster variant of `birsvd` that replaces the direct linear solvers with the
iterative LSQR algorithm, making it suitable for larger matrices.

Same parameters and return value as `birsvd`.

### `svd_imputation_with_mask(data, mask, n_rank, n_iter, ini_method='total_mean', velocity=0.1)`

Imputes missing values in `data` (indicated by zeros in `mask`) using iterative
randomized SVD.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `ndarray (m, n)` | Input data matrix |
| `mask` | `ndarray (m, n)` | Binary mask; 1 = observed, 0 = missing |
| `n_rank` | `int` | Rank of the SVD approximation |
| `n_iter` | `int` | Number of imputation iterations |
| `ini_method` | `str` | Initial fill strategy: `'total_mean'`, `'column_mean'`, `'row_mean'`, `'all_zeros'` |
| `velocity` | `float` | Update step size in (0, 1] |

Returns `(u, s, v)` — the SVD factors of the imputed matrix.

### `BIRSVDParameter`

Dataclass for configuring the `birsvd` and `birsvd_fast` algorithms.

| Attribute | Default | Description |
|-----------|---------|-------------|
| `n_iter` | `30` | Number of outer iterations |
| `init_method` | `'randOrthoNormal'` | Initialization strategy: `'randOrthoNormal'`, `'zeroOneVectors'`, `'polyOrthoNormal'` |
| `r_type_L` | `'2ndOrderDiff_acc8'` | Regularization type for the left factor |
| `r_type_R` | `'2ndOrderDiff_acc8'` | Regularization type for the right factor |
| `r_degree_L` | `0.0001` | Regularization strength for the left factor |
| `r_degree_R` | `0.0001` | Regularization strength for the right factor |
| `lsqr_niter` | `25` | LSQR inner iterations (used by `birsvd_fast` only) |

Available regularization types: `'TiKh'` (Tikhonov), `'2ndOrderDiff_acc2'`, `'2ndOrderDiff_acc4'`, `'2ndOrderDiff_acc6'`, `'2ndOrderDiff_acc8'`.

## Usage

```python
import numpy as np
from birsvd import birsvd, birsvd_fast, svd_imputation_with_mask
from birsvd.svd_settings import BIRSVDParameter

# Weighted low-rank approximation
m, n, r = 100, 80, 5
data   = np.random.randn(m, n)
weight = np.random.rand(m, n)   # values in (0, 1]

approx = birsvd(data, weight, n_rank=r)

# Faster variant for larger matrices
approx_fast = birsvd_fast(data, weight, n_rank=r)

# Custom parameters
param = BIRSVDParameter(n_iter=50, r_degree_L=1e-3, r_degree_R=1e-3)
approx = birsvd(data, weight, n_rank=r, param=param)

# Missing-value imputation
mask = (np.random.rand(m, n) > 0.2).astype(float)   # 20 % missing
u, s, v = svd_imputation_with_mask(data, mask, n_rank=r, n_iter=20)
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
