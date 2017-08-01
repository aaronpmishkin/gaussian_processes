# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:07:21
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-01 16:49:48

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
        var: number
            The variance magnitude of the kernel function.
    """

    def __init__(self, length_scale=1, var=1):
        self.length_scale = length_scale
        self.var = var

    def get_hyperparameters(self):
        return np.array([self.length_scale, self.var])

    def set_hyperparameters(self, theta):
        self.length_scale = theta[0]
        self.var = theta[1]

    def cov(self, X, Y=None, theta=None):
        """ cov
        Compute the covariance matrix of X and Y using the RBF kernel.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            Y (optional): array-like, shape = [m_samples, n_features]
                A second array of inputs.
                If Y is None, then the covariance matrix of X with itself will be computed.
            theta (optional): array-like, shape = [2, ]
                An array of hyperparameter values for the kernel.
        """
        if Y is None:
            Y = X

        if theta is None:
            theta = np.array([self.length_scale, self.var])

        # Compute a matrix of squared eucledian distances between X and Y
        dist = cdist(X, Y, 'sqeuclidean')

        K = theta[1] * np.exp(dist / (-2 * (theta[0] ** 2)))

        return K

    def cov_gradient(self, X, theta=None):
        """ cov_gradient
        Compute the gradient of the covariance matrix of X with respect to the hyperparameters
        of the RBF kernel.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            theta (optional): array-like, shape = [2, ]
                An array of hyperparameter values for the kernel.
        """

        if theta is None:
            theta = np.array([self.length_scale, self.var])

        dist = cdist(X, X, 'sqeuclidean')

        K = np.exp(dist / (-2 * (theta[0] ** 2)))

        dK_dl = theta[1] * (theta[0] ** -5) * dist * K

        dK_dvar = K

        return np.array([dK_dl, dK_dvar])


