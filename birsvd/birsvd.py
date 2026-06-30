#!/usr/bin/env python
# -*- coding: utf-8  -*-
from sklearn.utils.extmath import randomized_svd
import numpy as np
import scipy as sci

from .utils.parameter import BIRSVDParameter
from .utils.functions import __legendre_polys, __get_regularization_matrix
from .utils.results import SVDResult

DEFAULT_PARAM = BIRSVDParameter()


def __A_x(x, U, W, D):
    ''' Apply the LSQR forward operator.

    Arguments:
        x (ndarray):
            The vectorized right low-rank factor.
        U (ndarray):
            The left low-rank factor with shape (m, r).
        W (ndarray):
            The weight matrix with shape (m, n).
        D (ndarray):
            The regularization matrix with shape (n, n).

    Returns:
        ndarray:
            The concatenated data-fit and regularization components.
    '''
    m, n = W.shape
    r = U.shape[1]

    x_r = x.reshape((n, r)).T
    b_u = W * U.dot(x_r)
    b_l = np.dot(x_r, D.T)

    return np.concatenate(
        [b_u.T.flatten(), b_l.T.flatten()], axis=0)


def __Ap_x(x, U, W, D):
    ''' Apply the LSQR adjoint operator.

    Arguments:
        x (ndarray):
            The concatenated data-fit and regularization vector.
        U (ndarray):
            The left low-rank factor with shape (m, r).
        W (ndarray):
            The weight matrix with shape (m, n).
        D (ndarray):
            The regularization matrix with shape (n, n).

    Returns:
        ndarray:
            The vectorized adjoint product.
    '''
    m, n = W.shape
    r = U.shape[1]

    b_l = U.T.dot((W.T.flatten() * x[0:n * m]).reshape(n, m).T)
    x_r = x[n * m:].reshape((n, r))
    b_r = np.dot(D.T, x_r)

    return b_l.T.flatten() + b_r.flatten()


def least_square_low_rank(
        U, W, D, b, n_iter,
        A=__A_x, Ap=__Ap_x, x_init=None):
    ''' Solve a low-rank weighted least-squares problem with LSQR.

    This solves the least-squares problem ``min || A x - b ||`` that arises in
    weighted low-rank approximation. The forward and adjoint operators are
    supplied as function handles.

    Arguments:
        U (ndarray):
            The left low-rank factor with shape (m, r).
        W (ndarray):
            The weight matrix with shape (m, n).
        D (ndarray):
            The regularization matrix.
        b (ndarray):
            The right-hand-side vector.
        n_iter (int):
            The number of LSQR iterations.
        A (callable):
            The forward operator.
            default: __A_x
        Ap (callable):
            The adjoint operator.
            default: __Ap_x
        x_init (ndarray):
            The optional initial solution vector.
            default: None

    Returns:
        ndarray:
            The solution estimate at each LSQR iteration. The shape is
            (n_iter, n * r).
    '''

    if x_init is not None:
        b = b - A(x_init, U, W, D)
        x_aff = x_init

    beta = np.linalg.norm(b)
    u, phibar = b / beta, beta

    v = Ap(u, U, W, D)
    alpha = np.linalg.norm(v)
    v, rhobar = v / alpha, alpha
    w = v

    x0 = np.zeros(shape=v.shape)
    x  = np.zeros(shape=(n_iter, x0.size))

    for i in range(n_iter):
        # bidiagonalization
        u = A(v, U, W, D) - alpha * u
        beta = np.linalg.norm(u)
        u = u / beta

        v = Ap(u, U, W, D) - beta * v
        alpha = np.linalg.norm(v)
        v = v / alpha

        # next orthogonal trans.
        rho = np.sqrt(rhobar * rhobar + beta * beta)
        c = rhobar / rho
        s = beta / rho
        t = s * alpha

        rhobar = -c * alpha
        phi = c * phibar
        phibar = s * phibar

        # update x,w
        x0 = x0 + (phi / rho) * w
        w = v - (t / rho) * w
        x[i, :] = x0

    return x


def birsvd(data, weight, n_rank, param=DEFAULT_PARAM):
    ''' Compute a weighted low-rank approximation with iterative LSQR solves.

    This is a faster variant of ``birsvd_original`` for larger matrices. It
    alternately solves for the right and left low-rank factors using the
    matrix-free LSQR helper ``least_square_low_rank``.

    Arguments:
        data (ndarray):
            The input data matrix. The shape should be (m, n).
        weight (ndarray):
            The non-negative weight matrix. The shape should match ``data``.
        n_rank (int):
            The target rank of the approximation. The value should be between
            1 and ``min(data.shape)``.
        param (BIRSVDParameter):
            The algorithm parameters. ``param.n_iter`` controls both the
            number of outer iterations and the number of LSQR iterations for
            each factor update.
            default: DEFAULT_PARAM

    Returns:
        SVDResult:
            The SVD factors and per-iteration weighted errors.

    Raises:
        ValueError:
            If ``data`` and ``weight`` have different shapes.
    '''
    if data.shape != weight.shape:
        raise ValueError(
            'the shape of "data" is not matched to that of "weight".')

    m, n = data.shape
    if n_rank < 1 or n_rank > min(m, n):
        raise ValueError('"n_rank" should be between 1 and min(data.shape).')

    D_U = __get_regularization_matrix(m, param.r_type_L)
    D_V = __get_regularization_matrix(n, param.r_type_R)


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
    Y  = np.zeros(shape=(n, n_rank))
    X  = np.zeros(shape=(m, n_rank))

    error  = []
    S = np.zeros(shape=(n_rank,))
    V = np.zeros(shape=(n, n_rank))

    for i in range(param.n_iter):
        tmp = np.concatenate([dw.T.flatten(), np.zeros(shape=(n * n_rank))])
        Y_all = least_square_low_rank(
            U, weight, param.r_degree_R * D_V, tmp, param.n_iter)
        Y = Y_all[-1, :]
        Y = Y.reshape(n, n_rank).T

        x, S, V = randomized_svd(Y, n_rank)
        V = V.T

        tmp = np.concatenate([dw.flatten(), np.zeros(shape=(m * n_rank))])
        X_all = least_square_low_rank(
            V, weight.T, param.r_degree_L * D_U, tmp, param.n_iter)
        X = X_all[-1, :]
        X = X.reshape(m, n_rank).T

        U, x = np.linalg.qr(X.T, mode='reduced')

        approx = U.dot(np.diag(S).dot(V.T))

        err  = np.linalg.norm((data - approx) * weight)

        error.append(err)

    return SVDResult(U=U, S=S, V=V, error=np.array(error))
