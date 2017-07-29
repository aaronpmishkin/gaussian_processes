# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:42:40
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-07-28 22:08:02


import numpy as np


#  The default mean function of a Gaussian process.
def zero_mean(X):
    return np.zeros(X.shape)
