# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:42:40
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 15:14:12


import numpy as np


class Zero():
    """ Zero
    Zero mean function.
    ----------
        dim: integer
            The dimensionality of inputs to the mean function (i.e. dimension of X).
    """

    def __init__(self, dim=1):
        self.dim = dim

    def f(self, X):
        """ f
        Evaluates the mean function on an array of inputs X.
        Returns an array of 0's with shape [n_samples, 1]
        ----------
            X: array-like, shape = [n_samples, n_features]
                The array of inputs on which to evaluate the mean function.
        """
        return np.zeros((X.shape[0], 1))

    def f_grad(self, x):
        """ f_grad
        Evaluates the gradient of the zero mean function at x.
        This is always zero.
        ----------
            x: array-like, shape = [n_features, ]
                The input vector at which to evaluate the gradient.
        """
        return np.zeros((x.shape[0]))


class AdditiveMean():
    """ AdditiveMean
    An additive mean function where each feature has its own (independent) mean function.
    ----------
        dim: integer
            The dimensionality of inputs to the mean function (i.e. dimension of X).
        mean_functions: array-like, shape = [n_features, ]
            The array of mean functions, one for each feature/dimension.
    """

    def __init__(self, dim, mean_functions):
        self.dim = dim
        self.mean_functions = mean_functions

    def f(self, X):
        """ f
        Evaluates the mean function on an array of inputs X.
        ----------
            X: array-like, shape = [n_samples, n_features]
                The array of inputs on which to evaluate the mean function.
        """
        return np.array(list(map(self.__f__, X)))

    def __f__(self, x):
        """ f
        Evaluates the mean function on a single inputs x.
        This is an internal helper for f.
        ----------
            X: array-like, shape = [n_features, ]
                The single input on which to evaluate the mean function.
        """
        y = 0
        for key in self.mean_functions:
            y += self.mean_functions[key][0](x)
        return np.array([y])

    def f_grad(self, x):
        """ f_grad
        Evaluates the gradient of the mean function at x.
        ----------
            x: array-like, shape = [n_features, ]
                The input vector at which to evaluate the gradient.
        """
        gradient = []

        for key in self.mean_functions:
            gradient.push(self.mean_functions[key][1](x))

        return np.array(gradient)
