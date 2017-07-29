# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:07:21
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-07-28 22:31:57

import numpy as np
from scipy.spatial.distance import cdist

# k(x_i, x_j) = exp(-1 / 2 ||x_i - x_j|| / l)^2)


class RBF():
    """ RBF
    Implementation of the radial basis function kernel. Also called the squared exponential kernel.
    Arguments:
    ----------
        length_scale: number
            The length scale of the kernel function.
        sigma: number
            The variance magnitude of the kernel function.
    """

    def __init__(self, length_scale=1, sigma=1):
        self.length_scale = length_scale
        self.sigma = sigma

    def cov(self, X, Y=None):
        if Y is None:
            Y = X

        # Compute a matrix of squared eucledian distances between X and Y
        K = cdist(X, Y, 'sqeuclidean')

        K = self.sigma * np.exp(K / (-2 * (self.length_scale ** 2)))

        return K
