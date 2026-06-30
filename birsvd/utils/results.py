#!/usr/bin/env python
# -*- coding: utf-8  -*-
from dataclasses import dataclass

import numpy as np

__all__ = [
    'SVDResult'
]


@dataclass
class SVDResult:
    ''' Result of an SVD-based approximation.

    Attributes:
        U (ndarray):
            The left singular vectors.
        S (ndarray):
            The singular values.
        V (ndarray):
            The right singular vectors.
        error (ndarray):
            The per-iteration error values.
    '''
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray
    error: np.ndarray

    def __post_init__(self):
        if self.U.ndim != 2:
            raise ValueError('"U" should be a matrix.')

        if self.S.ndim != 1:
            raise ValueError('"S" should be a vector.')

        if self.V.ndim != 2:
            raise ValueError('"V" should be a matrix.')

        if self.U.shape[1] != self.S.size:
            raise ValueError(
                'The number of columns in "U" should match the size of "S".')

        if self.V.shape[1] != self.S.size:
            raise ValueError(
                'The number of columns in "V" should match the size of "S".')

        if self.error.ndim != 1:
            raise ValueError('"error" should be a vector.')

    @property
    def rank(self):
        ''' The target rank of the approximation '''
        return self.S.size

    @property
    def A(self):
        ''' The low-rank approximation matrix '''
        return self.U @ np.diag(self.S) @ self.V.T

    def residual(self, X):
        ''' The residual matrix of the approximation '''
        return X - self.A

    def loss(self, X, func=lambda x: np.sum(x ** 2)):
        ''' The loss value of the approximation '''
        return func(self.residual(X))
