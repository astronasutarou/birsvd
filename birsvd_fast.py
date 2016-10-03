#!/usr/bin/python
# -*- coding: utf-8  -*-

from __future__ import (
    absolute_import, division, print_function,unicode_literals)

import numpy as np
import scipy as sci
from progressbar import ProgressBar
from progressbar import Percentage, Bar
from sklearn.utils.extmath import randomized_svd

from birsvd import BIRSVDParameter as BIRSVDParameter
from birsvd import __get_regularization_matrix as __get_regularization_matrix

DEFAULT_PARAM = BIRSVDParameter()


def __A_x(x, U, W, D):
    m,n = W.shape
    r = U.shape[1]

    x_r = x.reshape((n, r)).T
    b_u = W*U.dot(x_r)
    b_l = np.dot(x_r,D)

    return np.concatenate(
        [b_u.T.flatten(), b_l.T.flatten()], axis=0)


def __Ap_x(x, U, W, D):
    m,n = W.shape
    r = U.shape[1]

    b_l = U.T.dot((W.T.flatten()*x[0:n*m]).reshape(n,m).T)
    x_r = x[n*m:].reshape((n, r))
    b_r = np.dot(D, x_r)

    return b_l.T.flatten() + b_r.T.flatten()


def least_square_low_rank(
        U, W, D, b, n_iter,
        A=__A_x, Ap=__Ap_x, x_init=None):
    u"""
    SYNOPSIS:

    solves the least squares problem  min || Ax-b || that arises in the
    "weighted low rank approximation with weighted data" problem for the
    vector x via the LSQR algorithm of Saunders and Paige.  numiter is
    the number of iterations to use. This implementation uses the operators
    A and A' as black boxes. Post-multipication of vectors with A
    or A' is acomplished by using the function handle.

    Input parameters:

    U           the left low rank approximants
    W           the weight matrix
    D           the matrix for regularization
    b           the vector on right hand side
    n_iter      the number of iterations
    A           function handle for post multipication Ax
    A_prime     function handle for post multipication A*x
    x0  first initial guess (optional argument, default set to 0)
    """

    if x_init is not None:
        b = b - A(x_init,U,W,D)
        x_aff = x_init

    beta = np.linalg.norm(b)
    u, phibar = b/beta, beta

    v = Ap(u,U,W,D)
    alpha = np.linalg.norm(v)
    v, rhobar = v/alpha, alpha
    w = v

    x0 = np.zeros(shape=v.shape)
    x  = np.zeros(shape=(n_iter, x0.size))

    for i in range(n_iter):
        # bidiagonalization
        u = A(v,U,W,D) - alpha*u
        beta = np.linalg.norm(u)
        u = u/beta

        v = Ap(u,U,W,D) - beta*v
        alpha = np.linalg.norm(v)
        v = v/alpha

        # next orthogonal trans.
        rho = np.sqrt(rhobar*rhobar + beta*beta)
        c = rhobar/rho
        s = beta/rho
        t = s*alpha

        rhobar = -c*alpha
        phi = c*phibar
        phibar = s*phibar

        # update x,w
        x0 = x0 + (phi/rho)*w
        w = v - (t/rho)*w
        x[i,:] = x0

    return x


def birsvd_fast(data, weight, n_rank, param=DEFAULT_PARAM):
    u"""
    """
    if data.shape != weight.shape:
        raise ValueError(
            'the shape of "data" is not matched to that of "weight".')

    m,n = data.shape

    D_U = __get_regularization_matrix(m, param.r_type_L)
    D_V = __get_regularization_matrix(n, param.r_type_R)


    # initialize
    if param.init_method == 'zeroOneVectors':
        U = np.zeros(shape=(m,n_rank))
        i = np.arange(1,n_rank+1)*int(m/n_rank)-1
        U[i,:] = np.eye(n_rank)
    elif param.init_method == 'randOrthoNormal':
        U = sci.linalg.orth(np.random.normal(size=(m, n_rank)))
    elif param.init_method == 'polyOrthoNormal':
        x = np.linspace(-1,1,m)
        U = __legendre_polys(n_rank, x)
    else:
        U = sci.linalg.orth(data[:,0:n_rank])

    dw = data * weight
    Y  = np.zeros(shape=(n, n_rank))
    X  = np.zeros(shape=(m, n_rank))

    lsqr_niter = param.lsqr_niter

    approx = np.zeros(shape=data.shape)
    error  = []

    progress = ProgressBar(widgets=[Percentage(), Bar()])
    for i in progress(range(param.n_iter)):

        tmp = np.concatenate([dw.T.flatten(),np.zeros(shape=(n*n_rank))])
        Y_all = least_square_low_rank(
            U, weight, param.r_degree_R * D_V, tmp, lsqr_niter)
        Y = Y_all[-1,:]
        Y = Y.reshape(n, n_rank).T

        x,S,V = randomized_svd(Y, n_rank)
        V = V.T

        tmp = np.concatenate([dw.flatten(),np.zeros(shape=(m*n_rank))])
        X_all = least_square_low_rank(
            V, weight.T, param.r_degree_L * D_U, tmp, lsqr_niter)
        X = X_all[-1,:]
        X = X.reshape(m, n_rank).T

        U,x = np.linalg.qr(X.T, mode='reduced')

        this = U.dot(np.diag(S).dot(V.T))

        err  = np.linalg.norm((data-this)*weight)
        errn = np.linalg.norm((data+this)*weight)
        if (err > errn): this,err = -this,errn

        tol = np.linalg.norm((approx-this))
        approx = this
        error.append(err)

    return approx
