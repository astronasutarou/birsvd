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
            options: 'zeroOneVectors', 'randOrthoNormal', and
                     'polyOrthoNormal'
            default: 'randOrthoNormal' (random orthonormal vectors)
        r_type_L (str):
            The type of regularization for the left singular vectors.
            options: 'TiKh', '2ndOrderDiff_acc2', '2ndOrderDiff_acc4',
                     '2ndOrderDiff_acc6', and '2ndOrderDiff_acc8'
            default: '2ndOrderDiff_acc8' (2nd order difference with accuracy 8)
        r_degree_L (float):
            The degree of regularization for the left singular vectors.
            default: 0.0001
        r_type_R (str):
            The type of regularization for the right singular vectors.
            options: 'TiKh', '2ndOrderDiff_acc2', '2ndOrderDiff_acc4',
                     '2ndOrderDiff_acc6', and '2ndOrderDiff_acc8'
            default: '2ndOrderDiff_acc8' (2nd order difference with accuracy 8)
        r_degree_R (float):
            The degree of regularization for the right singular vectors.
            default: 0.0001
    '''
    n_iter: int = 30
    init_method: str = 'randOrthoNormal'
    r_type_L: str = '2ndOrderDiff_acc8'
    r_degree_L: float = 0.0001
    r_type_R: str = '2ndOrderDiff_acc8'
    r_degree_R: float = 0.0001

    def __post_init__(self):
        valid_init_methods = [
            'zeroOneVectors',
            'randOrthoNormal',
            'polyOrthoNormal'
        ]
        valid_r_types = ['TiKh', '2ndOrderDiff_acc2', '2ndOrderDiff_acc4',
                         '2ndOrderDiff_acc6', '2ndOrderDiff_acc8']

        if self.init_method not in valid_init_methods:
            raise ValueError(
                f'Invalid init_method: {self.init_method}.'
                f'Valid options are: {valid_init_methods}')

        if self.r_type_L not in valid_r_types:
            raise ValueError(
                f'Invalid r_type_L: {self.r_type_L}. '
                f'Valid options are: {valid_r_types}')

        if self.r_type_R not in valid_r_types:
            raise ValueError(
                f'Invalid r_type_R: {self.r_type_R}. '
                f'Valid options are: {valid_r_types}')


DEFAULT_PARAM = BIRSVDParameter()
