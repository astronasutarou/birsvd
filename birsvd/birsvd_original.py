#!/usr/bin/env python
# -*- coding: utf-8  -*-
from sklearn.utils.extmath import randomized_svd
import numpy as np
import scipy as sci

from .svd.parameter import BIRSVDParameter
from .svd.functions import __legendre_polys, __get_regularization_matrix

DEFAULT_PARAM = BIRSVDParameter()


def birsvd_original(data, weight, n_rank, param=DEFAULT_PARAM):
    ''' Compute a weighted low-rank approximation with BIRSVD.

    This implementation alternately solves for the right and left low-rank
    factors using direct dense linear algebra.

    Arguments:
        data (ndarray):
            The input data matrix. The shape should be (m, n).
        weight (ndarray):
            The non-negative weight matrix. The shape should match ``data``.
        n_rank (int):
            The target rank of the approximation. The value should be between
            1 and ``min(data.shape)``.
        param (BIRSVDParameter):
            The algorithm parameters.
            default: DEFAULT_PARAM

    Returns:
        ndarray:
            The weighted low-rank approximation matrix with shape (m, n).

    Raises:
        ValueError:
            If ``data`` and ``weight`` have different shapes.
    '''
    if data.shape != weight.shape:
        raise ValueError('Shapes of "data" and "weight" should be the same.')

    m, n = data.shape
    if n_rank < 1 or n_rank > min(m, n):
        raise ValueError('"n_rank" should be between 1 and min(data.shape).')

    # get regularization matrices
    D_U = __get_regularization_matrix(m, param.r_type_L)
    D_V = __get_regularization_matrix(n, param.r_type_R)
    iDU = np.zeros(shape=(m * n_rank, m * n_rank))
    iDV = np.zeros(shape=(n * n_rank, n * n_rank))
    for r in range(n_rank):
        iDU[r::n_rank, r::n_rank] = D_U
        iDV[r::n_rank, r::n_rank] = D_V
    iDU = param.r_degree_L * iDU
    iDV = param.r_degree_R * iDV
    iDU = iDU.T.dot(iDU)
    iDV = iDV.T.dot(iDV)


    # initialize
    if param.init_method == 'zeroOneVectors':
        U = np.zeros(shape=(m, n_rank))
        i = np.arange(1, n_rank + 1) * int(m / n_rank) - 1
        U[i, :] = np.eye(n_rank)
    elif param.init_method == 'randOrthoNormal':
        U = sci.linalg.orth(np.random.normal(size=(m, n_rank)))
    elif param.init_method == 'polyOrthoNormal':
        x = np.linspace(-1, 1, m)
        U = __legendre_polys(n_rank, x)
    else:
        U = sci.linalg.orth(data[:, 0:n_rank])

    dw = data * weight
    approx = np.zeros(shape=data.shape)
    error = []

    for i in range(param.n_iter):
        R = (U.T.dot(dw)).T.flatten()
        L = np.zeros(iDV.shape)
        for j in range(n):
            L[j * n_rank:(j + 1) * n_rank, j * n_rank:(j + 1) * n_rank] \
                = U.T.dot(np.tile(weight[:, j].reshape(m, 1), (1, n_rank)) * U)
        L_L = np.linalg.cholesky(L + iDV)
        Y = np.linalg.lstsq(L_L.T, np.linalg.lstsq(L_L, R)[0])[0]

        Y = Y.T.reshape(n, n_rank).T

        x, S, V = randomized_svd(Y, n_rank)
        V = V.T

        R = (V.T.dot(dw.T)).T.flatten()
        L = np.zeros(iDU.shape)
        for j in range(m):
            L[j * n_rank:(j + 1) * n_rank, j * n_rank:(j + 1) * n_rank] \
                = V.T.dot(np.tile(weight[j, :].reshape(n, 1), (1, n_rank)) * V)
        L_L, U_L = sci.linalg.lu(L + iDU, permute_l=True)
        X = np.linalg.lstsq(U_L, np.linalg.lstsq(L_L, R)[0])[0]
        X = X.T.reshape(m, n_rank).T

        U, x = np.linalg.qr(X.T, mode='reduced')

        approx = U.dot(np.diag(S).dot(V.T))

        err  = np.linalg.norm((data - approx) * weight)

        error.append(err)

    return approx
