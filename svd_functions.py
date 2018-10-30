#!/usr/bin/env python
# -*- coding: utf-8  -*-
from __future__ import print_function, absolute_import, division
import numpy as np


def __legendre_polys(n_poly, x):
  if x.ndim!=1:
    raise ValueError('"x" should be a vector.')

  n_data = x.size
  y = np.zeros(shape=(n_data,n_poly+1))

  y[:,0] = np.ones(shape=(n_data,))
  y[:,1] = x

  for i in range(1,n_poly):
    tmp = (2*i-1)*x*y[:,i]-(i-1)*y[:,i-1]
    y[:,i+1] = tmp/i

  return y[:,0:n_poly]


def __get_regularization_matrix(n, r_type):
  if r_type == 'TiKh':
    D = np.eye(n)
  elif r_type == '2ndOrderDiff_acc2':
    D = np.diag(  (1./2)*np.ones(shape=(n-1,)), 1) \
      + np.diag((-1.0)*np.ones(shape=(n-0,)), 0) \
      + np.diag((1./2)*np.ones(shape=(n-1,)),-1)
    D[0,1] = 1.0
    D[n-1,n-2] = 1.0
  elif r_type == '2ndOrderDiff_acc4':
    D = np.diag(  (-1./12)*np.ones(shape=(n-2,)), 2) \
      + np.diag(  (4./3)*np.ones(shape=(n-1,)), 1) \
      + np.diag( (-5./2)*np.ones(shape=(n-0,)), 0) \
      + np.diag(  (4./3)*np.ones(shape=(n-1,)),-1) \
      + np.diag((-1./12)*np.ones(shape=(n-2,)),-2)
    D[0,0:3] = (-5./2, 8./3, -1./6)
    D[1,0:4] = (4./3, -5./2, 4./3, -1./6)
    D[(n-3):n,n-1] = (-5./2, 8./3, -1./6)
    D[(n-4):n,n-2] = (4./3, -5./2, 4./3, -1./6)
  elif r_type == '2ndOrderDiff_acc6':
    D = np.diag(  (1./90)*np.ones(shape=(n-3,)), 3) \
      + np.diag( (-3./20)*np.ones(shape=(n-2,)), 2) \
      + np.diag(   (3./2)*np.ones(shape=(n-1,)), 1) \
      + np.diag((-49./18)*np.ones(shape=(n-0,)), 0) \
      + np.diag(   (3./2)*np.ones(shape=(n-1,)),-1) \
      + np.diag(  (3./20)*np.ones(shape=(n-2,)),-2) \
      + np.diag( (-1./90)*np.ones(shape=(n-3,)),-3)
    D[0,0:4] = (-49./18, 3, -3./10, 1./45)
    D[1,0:5] = (3./2, -49./18, 3./2, -3./10, 1./45)
    D[2,0:6] = (-3./20, 3./2, -49./18, 3./2, -3./20, 1./45)
    D[(n-4):n,n-1] = (-49./18, 3, -3./10, 1./45)
    D[(n-5):n,n-2] = (3./2, -49./18, 3./2, -3./10, 1./45)
    D[(n-6):n,n-3] = (-3./20, 3./2, -49./18, 3./2, -3./20, 1./45)
  elif r_type == '2ndOrderDiff_acc8':
    D = np.diag(   (-1./560)*np.ones(shape=(n-4,)), 4) \
      + np.diag(  (8./315)*np.ones(shape=(n-3,)), 3) \
      + np.diag(   (-1./5)*np.ones(shape=(n-2,)), 2) \
      + np.diag(  (8./5)*np.ones(shape=(n-1,)), 1) \
      + np.diag((-205./72)*np.ones(shape=(n-0,)), 0) \
      + np.diag(  (8./5)*np.ones(shape=(n-1,)),-1) \
      + np.diag(   (-1./5)*np.ones(shape=(n-2,)),-2) \
      + np.diag(  (8./315)*np.ones(shape=(n-3,)),-3) \
      + np.diag( (-1./560)*np.ones(shape=(n-4,)),-4)
    D[0,0:5] = (-205./72, 16./5, -2./5, 16./315, -1./280)
    D[1,0:6] = (8./5, -205./72, 8./5, -2./5, 16./315, -1./280)
    D[2,0:7] = (-1./5, 8./5, -205./72, 8./5, -1./5, 16./315, -1./280)
    D[3,0:8] = (8./315, -1./5, 8./5, -205./72,
          8./5, -1./5, 8./315, -1./280)
    D[(n-5):n,n-1] = (-205./72, 16./5, -2./5, 16./315, -1./280)
    D[(n-6):n,n-2] = (8./5, -205./72, 8./5, -2./5, 16./315, -1./280)
    D[(n-7):n,n-3] = (-1./5, 8./5, -205./72, 8./5, -1./5, 16./315, -1./280)
    D[(n-8):n,n-4] = (8./315, -1./5, 8./5, -205./72,
              8./5, -1./5,8./315, -1./280)
  else:
    raise ValueError('Wrong "r_type" is assigned.')

  return D
