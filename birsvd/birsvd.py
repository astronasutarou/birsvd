#!/usr/bin/env python
# -*- coding: utf-8  -*-
from sklearn.utils.extmath import randomized_svd
from birsvd.svd_settings import BIRSVDParameter
from birsvd.svd_functions import __get_regularization_matrix
import numpy as np
import scipy as sci

DEFAULT_PARAM = BIRSVDParameter()


def birsvd(data, weight, n_rank, param=DEFAULT_PARAM):
    if data.shape != weight.shape:
        raise ValueError('Shapes of "data" and "mask" should be the same.')

    m, n = data.shape

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

        this = U.dot(np.diag(S).dot(V.T))

        err  = np.linalg.norm((data - this) * weight)
        errn = np.linalg.norm((data + this) * weight)
        if err > errn:
            this, err = -this, errn

        tol = np.linalg.norm((approx - this))
        approx = this
        error.append(err)

    return approx
