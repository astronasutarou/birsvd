#!/usr/bin/env python
# -*- coding: utf-8  -*-
from dataclasses import dataclass

__all__ = [
    'BIRSVDParameter'
]


@dataclass(frozen=True)
class BIRSVDParameter:
    ''' Parameters for BIRSVD algorithm

    Attributes:
        n_iter (int):
            The number of iterations for the whole algorithm.
            default: 30
        init_method (str):
            The method for initializing the left singular vectors.
            default: 'randOrthoNormal' (random orthonormal vectors)
        r_type_L (str):
            The type of regularization for the left singular vectors.
            default: '2ndOrderDiff_acc8' (2nd order difference with accuracy 8)
        r_degree_L (float):
            The degree of regularization for the left singular vectors.
            default: 0.0001
        r_type_R (str):
            The type of regularization for the right singular vectors.
            default: '2ndOrderDiff_acc8' (2nd order difference with accuracy 8)
        r_degree_R (float):
            The degree of regularization for the right singular vectors.
            default: 0.0001
        lsqr_niter (int):
            The number of iterations for the LSQR algorithm.
            default: 25
    '''
    n_iter: int = 30
    init_method: str = 'randOrthoNormal'
    r_type_L: str = '2ndOrderDiff_acc8'
    r_degree_L: float = 0.0001
    r_type_R: str = '2ndOrderDiff_acc8'
    r_degree_R: float = 0.0001
    lsqr_niter: int = 25

DEFAULT_PARAM = BIRSVDParameter()
