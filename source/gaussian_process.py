# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 16:07:12
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-07-28 23:01:52

# Implementation adapted from Gaussian Processes for Machine Learning; Rasmussen and Williams, 2006

import numpy as np
from scipy.linalg import cholesky, inv, det
from scipy.stats import norm

import mean_functions

# Not that that this class currently assumes a Gaussian likelihood for data.
# TODO: Implement the log marginal likelihood function


class GaussianProcess():

    """ GaussianProcess
    Gaussian Process for non-parametric regression.
    Arguments:
    ----------
        X: array-like, shape = [n_samples, n_features]
            The inputs to the Gaussian Process
        y: array-like, shape = [n_samples, 1]
            The targets for the Gaussian Process
        kernel:
            The kernel function for the Gaussian Process.
            This expresses the covariance between inputs.
        mean_function:
            The prior mean function for the gaussian process.
        sigma: number
            The variance of the gaussian noise in the targets.
    """

    def __init__(self, X, Y, kernel, mean_function=mean_functions.zero_mean, sigma=1):
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.mean_function = mean_function
        self.mu = mean_function(X)
        self.sigma = sigma

        self.K = kernel.cov(X) + ((sigma ** 2) * np.identity(X.shape[0]))

    def predict(self, X_star, noise=True):
        """ predict
        Predict the targets for a list of inputs
        Arguments:
        ----------
            X_star: array-like, shape = [n_samples, n_features]
                The inputs to for which to predict targets.
            noise: Boolean
                Whether or not to include noise estimate in predictions.
        """
        K_star = self.kernel.cov(X_star, self.X)

        L = cholesky(self.K, lower=True)
        L_inv = inv(L)

        alpha = np.dot(L_inv.T, np.dot(L_inv, (self.Y - self.mu)))

        f_bar = self.mean_function(X_star) + np.dot(K_star, alpha)

        cov = self.kernel.cov(X_star) - np.dot(np.dot(K_star, L_inv.T), np.dot(L_inv, K_star.T))

        if noise:
            cov += (self.sigma ** 2) * np.identity(cov.shape[0])

        return f_bar, cov  # Return the mean and covariance matrix for X_star

    def predict_quantiles(self, X_star, noise=True, confidence_bounds=(0.025, 0.975)):
        """ predict_quantiles
        Predict the upper and lower confidence bounds for the given targets.
        Arguments:
        ----------
            X_star: array-like, shape = [n_samples, n_features]
                The inputs for which to predict confidence bounds.
            noise: Boolean
                Whether or not to include noise estimate in predictions.
            confidence_bounds: tuple (lower_bound, upper_bound)
                The lower and upper quantiles that define the desired confidence bounds.
        """
        f_bar, cov = self.predict(X_star, noise)
        variance = np.diag(cov)
        upper_quantiles = np.copy(f_bar)
        lower_quantiles = np.copy(f_bar)

        alpha = confidence_bounds[1] - confidence_bounds[0]

        # Assuming that the likelihood is Gaussian:
        for i, mean in enumerate(f_bar):
            quantiles = norm.interval(alpha, loc=mean, scale=variance[i])
            lower_quantiles[i] = quantiles[0]
            upper_quantiles[i] = quantiles[1]

        return upper_quantiles, lower_quantiles

    def log_likelihood(self):
        """ log_likelihood
        Compute the log of the marginal likelihood of the model given the training inputs and targets.
        """
        L = cholesky(self.K, lower=True)
        L_inv = inv(L)

        Y_cent = self.Y - self.mu

        fit_term = np.log(np.dot(Y_cent.T, np.dot(np.dot(L_inv.T, L_inv), Y_cent)))

        complexity_term = np.log(det(self.K))

        normalizing_term = self.X.shape[0] * np.log(2 * np.pi)

        return (-1 / 2) * (fit_term + complexity_term + normalizing_term)

