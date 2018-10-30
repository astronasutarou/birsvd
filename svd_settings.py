#!/usr/bin/env python
# -*- coding: utf-8  -*-
from __future__ import print_function, absolute_import, division
import numpy as np


class BIRSVDParameter(object):
  def __init__(self, n_iter=30, init_method='randOrthoNormal',
         r_type_L = '2ndOrderDiff_acc8', r_degree_L  = 0.001,
         r_type_R = '2ndOrderDiff_acc8', r_degree_R  = 0.001,
         lsqr_niter = 25):
    super(BIRSVDParameter,self).__init__()
    self.n_iter = n_iter
    self.init_method = init_method
    self.r_type_L = r_type_L
    self.r_type_R = r_type_R
    self.r_degree_L = r_degree_L
    self.r_degree_R = r_degree_R
    self.lsqr_niter = lsqr_niter

DEFAULT_PARAM = BIRSVDParameter()
