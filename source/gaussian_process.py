# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 16:07:12
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-01 16:49:48

# Implementation adapted from Gaussian Processes for Machine Learning; Rasmussen and Williams, 2006

import numpy as np
from scipy.linalg import cholesky, inv, solve_triangular
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
        obs_variance: number
            The variance of the gaussian noise in the targets.
    """

    def __init__(self, X, Y, kernel, mean_function=mean_functions.zero_mean, obs_variance=1):
        self.__plot_density__ = 200
        self.__plot_delta__ = 0.2

        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.mean_function = mean_function
        self.mu = mean_function(X)
        self.obs_variance = obs_variance

        self.K = kernel.cov(X) + (obs_variance * np.identity(X.shape[0]))

    def get_hyperparameters(self):
        return np.append(np.array([self.obs_variance]), self.kernel.get_hyperparameters())

    def set_hyperparameters(self, theta):
        self.obs_variance = theta[0]
        self.kernel.set_hyperparameters(theta[1:])
        self.K = self.kernel.cov(self.X) + (self.obs_variance * np.identity(self.X.shape[0]))

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

        alpha = solve_triangular(L, (self.Y - self.mu), lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        f_bar = self.mean_function(X_star) + np.dot(K_star, alpha)

        v = solve_triangular(L, K_star.T, lower=True)
        cov = self.kernel.cov(X_star) - np.dot(v.T, v)

        if noise:
            cov += self.obs_variance * np.identity(cov.shape[0])

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
        sd = np.sqrt(np.diag(cov))
        upper_quantiles = np.copy(f_bar)
        lower_quantiles = np.copy(f_bar)

        alpha = confidence_bounds[1] - confidence_bounds[0]

        # Assuming that the likelihood is Gaussian:
        for i, mean in enumerate(f_bar):
            quantiles = norm.interval(alpha, loc=mean, scale=sd[i])
            lower_quantiles[i] = quantiles[0]
            upper_quantiles[i] = quantiles[1]

        return lower_quantiles, upper_quantiles

    def log_likelihood(self, theta=None):
        """ log_likelihood
        Compute the log of the marginal likelihood given the training inputs and targets.
        """
        I_matrix = np.identity(self.X.shape[0])

        if theta is None:
            theta = np.array([self.obs_variance, self.kernel.length_scale, self.kernel.var])
            K = self.K
        else:
            K = self.kernel.cov(self.X, theta=theta[1:]) + (theta[0] * I_matrix)

        Y_cent = self.Y - self.mu
        L = cholesky(K, lower=True)

        alpha = solve_triangular(L, Y_cent, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)

        fit_term = -0.5 * np.dot(Y_cent.T, alpha)

        complexity_term = -np.log(np.diag(L)).sum()

        normalizing_term = -0.5 * self.X.shape[0] * np.log(2 * np.pi)

        return (fit_term + complexity_term + normalizing_term).sum()

    def log_likelihood_gradient(self, theta=None):
        # The first element of the hyperparameter vector is the observation variance.

        if theta is None:
            theta = np.array([self.obs_variance, self.kernel.length_scale, self.kernel.var])

        I_matrix = np.identity(self.X.shape[0])
        grad = np.copy(theta)

        K = self.kernel.cov(self.X, theta=theta[1:]) + (theta[0] * I_matrix)

        L = cholesky(K, lower=True)
        L_inv = inv(L)
        alpha = solve_triangular(L, (self.Y - self.mu), lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        beta = np.dot(alpha, alpha.T) - np.dot(L_inv.T, L_inv)

        grad[0] = 0.5 * np.trace(beta)

        dK_dtheta = self.kernel.cov_gradient(self.X, theta=theta[1:])

        for i, dK_dh in enumerate(dK_dtheta):
            grad[i + 1] = 0.5 * np.trace(np.dot(beta, dK_dh))

        return grad

    def __objective__(self, theta=None):
        return -1 * self.log_likelihood(theta)

    def __objective_grad__(self, theta=None):
        return -1 * self.log_likelihood_gradient(theta)

    def optimize(self, n_restarts=10):
        best_theta = None
        min_objective = np.inf

        for start in np.random.uniform(np.array([0, 0, 0]),
                                       np.array([10, 10, 10]),
                                       size=(n_restarts, 3)):

            res = minimize(self.__objective__,
                           x0=start,
                           method="L-BFGS-B",
                           jac=self.__objective_grad__,
                           bounds=np.array([(1e-8, None), (1e-8, None), (0, None)])
                           )

            if res.fun < min_objective:
                min_objective = res.fun
                best_theta = res.x

        return best_theta, -1 * min_objective

    def plot(self, show_data=True):
        if self.X.shape[1] != 1:
            raise ValueError('Cannot plot with multi-dimensional inputs')

        minVal = np.min(self.X)
        maxVal = np.max(self.X)
        delta = (maxVal - minVal) * self.__plot_delta__

        mean_x = np.linspace(np.floor(minVal - delta),
                             np.ceil(maxVal + delta),
                             self.__plot_density__).reshape(self.__plot_density__, 1)

        mean_y, cov = self.predict(mean_x)
        lq, uq = self.predict_quantiles(mean_x, noise=True)

        mean_x = mean_x.reshape(self.__plot_density__)

        fig = plt.figure()
        plt.plot(mean_x, mean_y, 'k', label='Predicted Mean')

        if show_data:
            plt.plot(self.X, self.Y, 'rx', label='Observed Inputs')

        plt.fill_between(mean_x, lq.reshape(lq.shape[0],),
                         uq.reshape(lq.shape[0],),
                         color='#B7E9F9', label='Confidence Bound')

        plt.legend()

        return fig

