#!/usr/bin/env python
# -*- coding: utf-8  -*-
from sklearn.utils.extmath import randomized_svd
import numpy as np


def svd_imputation_with_mask(
        data, mask, n_rank, n_iter,
        ini_method='total_mean',
        velocity=0.1):
    if data.shape != mask.shape:
        raise ValueError('Shapes of "data" and "mask" should be the same.')

    m, n = data.shape

    if ini_method == 'column_mean':
        cm = np.sum(data * mask, axis=0) / np.sum(mask, axis=0)
        data = data * mask + (1 - mask) * cm
    elif ini_method == 'row_mean':
        rm = np.sum(data * mask, axis=1) / np.sum(mask, axis=1)
        data = data * mask + ((1 - mask).T * rm).T
    elif ini_method == 'total_mean':
        tm = np.sum(data * mask) / np.sum(mask)
        data = data * mask + (1 - mask) * tm
    elif ini_method == 'all_zeros':
        data = data * mask
    else:
        raise ValueError('Invalid "ini_method" is given.')

    u, s, v = randomized_svd(data, n_rank)

    for i in range(n_iter):
        data[mask == 0] = (1 - velocity) * data[mask == 0] \
            + velocity * (u.dot(np.diag(s)).dot(v))[mask == 0]
        u, s, v = randomized_svd(data, n_rank)

    return u, s, v
