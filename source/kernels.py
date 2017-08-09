# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:07:21
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-08 20:17:01

import numpy as np
from scipy.spatial.distance import cdist


class RBF():
    """ RBF
    Implementation of the radial basis function kernel. Also called the squared exponential kernel.
    Arguments:
    ----------
        dim: integer
            The dimensionality of inputs to the kernel (i.e. dimension of X).
        length_scale: number
            The length scale of the kernel function.
        var: number
            The variance magnitude of the kernel function.
    """

    def __init__(self, dim, length_scale=1, var=1):
        self.dim = dim
        self.length_scale = length_scale
        self.var = var
        self.num_parameters = 2

    def get_parameters(self):
        """ get_parameters
        Get the kernel's parameters.
        """
        return np.array([self.length_scale, self.var])

    def set_parameters(self, theta):
        """ set_parameters
        Set the kernel's parameters.
        Arguments:
        ----------
            theta: array-like, shape = [2, ]
                An array containing the new parameters of the kernel.
                The parameter order is [length_scale, variance]
        """
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
                An array of parameter values for the kernel.
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
        Compute the gradient of the covariance matrix of X with respect to the parameters
        of the RBF kernel.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            theta (optional): array-like, shape = [2, ]
                An array of parameter values for the kernel.
        """

        if theta is None:
            theta = np.array([self.length_scale, self.var])

        dist = cdist(X, X, 'sqeuclidean')

        K = np.exp(dist / (-2 * (theta[0] ** 2)))

        dK_dl = theta[1] * (theta[0] ** -3) * dist * K

        dK_dvar = K

        return np.array([dK_dl, dK_dvar])


class Additive():
    """ RBF
    Implementation of the additive kernel as described by Duvenaud et al, 2011
    Arguments:
    ----------
        dim: integer,
            The dimensionality of inputs to the kernel (i.e. dimension of X).
        order: number, order <= dim
            The order of the additive kernel.
        base_kernels: array-like, shape = [dim, ]
            The set of base kernel functions, one for each dimension.
        var: array-like, shape = [order, ]
            An array of variance magnitudes, one for each order d: 1 <= d <= D
    """
    def __init__(self, dim, order, base_kernels, var=None):
        if order > dim:
            raise ValueError('Kernel order cannot be larger than input dimension')

        if dim != len(base_kernels):
            raise ValueError('A base kernel must be provided for each input dimension')

        if var is None:
            var = np.ones(dim)

        self.dim = dim
        self.order = order
        self.base_kernels = base_kernels
        self.var = var
        self.theta = self.get_parameters()
        self.num_parameters = len(self.theta)

    def get_parameters(self):
        """ get_parameters
        Get the kernel's parameters, which include the parameters of the base kernels.
        """
        theta = np.copy(self.var)

        for kernel in self.base_kernels:
            theta = np.append(theta, kernel.get_parameters())

        return theta

    def set_parameters(self, theta):
        """ set_parameters
        Set the kernel's parameters. This must include the parameters of the base kernels.
        Arguments:
        ----------
            theta: array-like, shape = [n_parameter, ]
                An array containing the new parameters of the kernel.
                The first |self.order| elements must be the interaction variance parameters.
                The remaining elements must be parameters for the base kernels.
        """
        self.var = theta[0:self.order]
        param_index = self.order

        for kernel in self.base_kernels:
            kernel.set_parameters(theta[param_index:param_index + kernel.num_parameters])
            param_index += kernel.num_parameters

    def __cov__(self, X, Y=None, order=None, theta=None, base_kernels=None):
        """ __cov__
        Compute the covariance matrix of inputs X and Y. Returns both the covariance matrix
        and a list of covariance matrices for each order of interaction.
        This is an internal helper. To obtain the just covariance matrix of X (and Y), call "cov"
        instead.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            Y (optional): array-like, shape = [m_samples, n_features]
                A second array of inputs.
            theta: array-like, shape = [n_parameter, ]
                The kernel parameters to use when computing the covariance.
                If None, the current parameters of the kernel are used.
            order: integer
                The interaction order that will be used.
                If None, the current kernel setting will be used.
            base_kernels: array-like, shape = [n_features, ]
                The list of base_kernels, one for each feature.
                Exactly one base_kernel must be provided per input feature.
                If None, the current base_kernels of the kernel are used.
        """
        if Y is None:
            Y = X

        if theta is None:
            theta = self.theta

        if order is None:
            order = self.order

        if base_kernels is None:
            base_kernels = self.base_kernels

        # Z is the array of covariance matrices produced by application of the base kernels.
        Z = np.ones((len(base_kernels), X.shape[0], Y.shape[0]))
        # S is the array of of k^th power sums of the matrices in Z, k = 1 ... self.order
        S = np.ones((order + 1, X.shape[0], Y.shape[0]))
        # K is the array of k^th order additive kernels, k = 1 ... order
        K = np.zeros((order + 1, X.shape[0], Y.shape[0]))
        K[0] = 1

        p_index = len(base_kernels)
        for i, kernel in enumerate(base_kernels):
            params = theta[p_index:p_index + kernel.num_parameters]
            p_index += kernel.num_parameters
            Z[i] = kernel.cov(X[:, i].reshape(X.shape[0], 1),
                              Y[:, i].reshape(Y.shape[0], 1),
                              theta=params)

        Z_d = np.copy(Z)
        for d in range(1, order + 1):
            S[d] = np.sum(Z_d, axis=0)
            Z_d = Z_d * Z_d

        for d in range(1, order + 1):
            for j in range(1, d + 1):
                K[d] += ((-1) ** (j - 1)) * K[d - j] * S[j]

            K[d] = K[d] / d

        for d in range(1, order + 1):
            K[d] = theta[d - 1] * K[d]

        return np.sum(K[1:], axis=0), K[1:]

    def cov(self, X, Y=None, order=None, theta=None, base_kernels=None):
        """ cov
        Compute the covariance matrix of inputs X and Y using __cov__.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            Y (optional): array-like, shape = [m_samples, n_features]
                A second array of inputs.
            theta: array-like, shape = [n_parameter, ]
                The kernel parameters to use when computing the covariance.
                If None, the current parameters of the kernel are used.
            order: integer
                The interaction order that will be used.
                If None, the current kernel setting will be used.
            base_kernels: array-like, shape = [X.shape[0], ]
                The base_kernels to use for each feature.
                Exactly one base_kernel must be provided per input feature.
                If None, the current base_kernels of the kernel are used.
        """
        K, K_orders = self.__cov__(X, Y, order, theta, base_kernels)

        return K

    def cov_gradient(self, X, theta=None):
        """ cov_gradient
        Compute the gradient of the covariance matrix of X with respect to the parameters
        of the additive kernel and the base kernels.
        Arguments:
        ----------
            X: array-like, shape = [n_samples, n_features]
                An array of inputs.
            theta: array-like, shape = [n_parameter, ]
                The kernel parameters to use when computing the covariance.
                If None, the current parameters of the kernel are used.
        """
        if theta is None:
            theta = self.theta

        gradient = []
        p_index = self.dim

        K, K_orders = self.__cov__(X, theta=theta)

        for i in range(self.order):
            gradient.append(K_orders[i])

        for i, ki in enumerate(self.base_kernels):
            dK_dki = self.cov(np.delete(X, i, axis=1),
                              order=(self.order - 1),
                              base_kernels=np.delete(self.base_kernels, i))

            dki_dtheta = ki.cov_gradient(X, theta[p_index: p_index + ki.num_parameters])

            gradient.extend((dK_dki + 1) * dki_dtheta)

        return np.array(gradient)







