# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 16:07:12
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-08 22:06:16

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
        Y: array-like, shape = [n_samples, 1]
            The targets for the Gaussian Process
        kernel:
            The kernel function for the Gaussian Process.
            This expresses the covariance between inputs.
        mean_function:
            The prior mean function for the gaussian process.
        obs_variance: number
            The variance of the gaussian noise in the targets.
    """

    def __init__(self, X, Y, kernel, mean_function=mean_functions.Zero(), obs_variance=1):
        self.__plot_density__ = 200
        self.__plot_delta__ = 0.2

        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.mean_function = mean_function
        self.mu = mean_function.f(X)
        self.Y_cent = self.Y - self.mu

        self.theta = np.append([obs_variance], self.kernel.get_parameters())
        _, self.K, self.L, self.alpha, _ = self.__covariance_at_theta__(self.theta)

    def get_hyperparameters(self):
        """ get_hyperparameters
        Get the hyperparameters of the Gaussian process model.
        """

        return self.theta

    def set_hyperparameters(self, theta):
        """ get_hyperparameters
        Set the hyperparameters of the Gaussian process model.
        Includes the arameters of the kernel function.
        Arguments:
        ----------
            Theta: array-like, shape = [n_hyperparameters, ]
                The new hyperparameters of the GP model. Must include kernel parameters.
        """
        self.theta = theta
        self.kernel.set_parameters(theta[1:])
        _, self.K, self.L, self.alpha, _ = self.__covariance_at_theta__(theta)

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

        f_bar = self.mean_function.f(X_star) + np.dot(K_star, self.alpha)

        v = solve_triangular(self.L, K_star.T, lower=True)
        cov = self.kernel.cov(X_star) - np.dot(v.T, v)

        if noise:
            cov += self.theta[0] * np.identity(cov.shape[0])

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
        Arguments:
        ---------
            theta: array-like, shape = [n_hyperparameters, ]
                The hyperparameter assignment at which to evaluate the log marginal likelihood.
        """
        theta, K, L, alpha, I_matrix = self.__covariance_at_theta__(theta)

        fit_term = -0.5 * np.dot(self.Y_cent.T, alpha)
        complexity_term = -np.log(np.diag(L)).sum()
        normalizing_term = -0.5 * self.X.shape[0] * np.log(2 * np.pi)

        return (fit_term + complexity_term + normalizing_term).sum()

    def log_likelihood_gradient(self, theta=None):
        """ log_likelihood_gradient
        Compute the gradient of log of the marginal likelihood
        with respect to the model hyperparameters.
        Arguments:
        ---------
            theta: array-like, shape = [n_hyperparameters, ]
                The hyperparameter assignment at which to evaluate the gradient.
        """
        theta, K, L, alpha, I_matrix = self.__covariance_at_theta__(theta)

        grad = np.copy(theta)
        L_inv = inv(L)
        beta = np.dot(alpha, alpha.T) - np.dot(L_inv.T, L_inv)
        grad[0] = 0.5 * np.trace(beta)
        dK_dtheta = self.kernel.cov_gradient(self.X, theta=theta[1:])

        for i, dK_dh in enumerate(dK_dtheta):
            grad[i + 1] = 0.5 * np.trace(np.dot(beta, dK_dh))

        return grad

    def __covariance_at_theta__(self, theta):
        """ __covariance_at_theta__
        Compute the covariance matrix of X at the hyperparameter assignment given by theta.
        If theta is None, the current hyperparameter assignment of the GP + kernel is used.
        Arguments:
        ---------
            theta: array-like, shape = [n_hyperparameters, ]
                The hyperparameter assignment at which to evaluate the covariance matrix.
        Return Type:
        ------------
            theta: array-like, shape = [n_hyperparameters, ]
                The hyperparameter assignment.
            K: array-like, shape = [n_samples, n_samples]
                The covariance matrix of X computed at theta.
            L: array-like, shape = [n_samples, n_samples]
                The cholesky (lower triangular) decomposition of K
            alpha: array-like, shape = [n_samples, ]
                alpha = inv(K)(Y_cent), the matrix product of the inverse of K
                and Y_cent, the centered observations.
            I_matrix: array-like, shape = [n_samples, n_samples]
                The identity matrix of size n by n.
        """
        I_matrix = np.identity(self.X.shape[0])

        if theta is None:
            theta = self.theta
            K = self.K
            L = self.L
            alpha = self.alpha
        else:
            K = self.kernel.cov(self.X, theta=theta[1:]) + (theta[0] * I_matrix)
            L = cholesky(K, lower=True)
            alpha = solve_triangular(L, self.Y_cent, lower=True)
            alpha = solve_triangular(L.T, alpha, lower=False)

        return theta, K, L, alpha, I_matrix

    def __objective__(self, theta=None, fixed_params=[]):
        """ __objective__
        Compute the negative log of the marginal likelihood given the training inputs and targets.
        This is the objective function used to optimize the hyperparameters of the GP model.
        Arguments:
        ---------
            theta: array-like, shape = [n_params, ]
                The (partial) hyperparameter assignment at which to evaluate the
                log marginal likelihood. The assignment is missing those hyperparams
                that are fixed in the optimization (see fixed_params).
            fixed_params: array-like, shape = [n_fixed_params, ], n_fixed_params <= n_hyperparams
                The indices of the hyperparameters in the hyperparameter array (theta)
                to consider fixed during optimization.
        """
        theta = np.insert(theta, fixed_params, self.theta[fixed_params])
        return -1 * self.log_likelihood(theta)

    def __objective_grad__(self, theta=None, fixed_params=[]):
        """ __objective_grad__
        Compute the negative gradient of the likelihood given the training inputs and targets.
        This is the gradient function used to optimize the hyperparameters of the GP model.
        Arguments:
        ---------
            theta: array-like, shape = [n_params, ]
                The (partial) hyperparameter assignment at which to evaluate the
                log marginal likelihood. The assignment is missing those hyperparams
                that are fixed in the optimization (see fixed_params).
            fixed_params: array-like, shape = [n_fixed_params, ], n_fixed_params <= n_hyperparams
                The indices of the hyperparameters in the hyperparameter array (theta)
                to consider fixed during optimization.
        """
        theta = np.insert(theta, fixed_params, self.theta[fixed_params])
        grad = -1 * self.log_likelihood_gradient(theta)

        return np.delete(grad, fixed_params)

    def optimize(self, bounds=None, n_restarts=10, fixed_params=[]):
        """ optimize
        Optimize the hyperparameters of the GP model using the marginal log likelihood.
        Arguments:
        ---------
            bounds: array-like, shape = [2, n_hyperparameters]
                The bounds to optimize the hyperparameters within.
                The first row contains lower bounds; the second upper bounds.
            n_restarts: integer
                The number of random restarts to perform during optimization.
            fixed_params: array-like, shape = [n_fixed_params, ], n_fixed_params <= n_hyperparams
                The indices of the hyperparameters in the hyperparameter array (theta)
                to consider fixed during optimization.
        """
        best_theta = None
        min_objective = np.inf

        if bounds is None:
            bounds = np.ones((self.kernel.num_parameters + 1, 2))
            bounds[:, 0] = bounds[:, 0] * 0.01
            bounds[:, 1] = bounds[:, 1] * 1000

        if len(fixed_params) != 0:
            bounds = np.delete(bounds, fixed_params, axis=0)

        for start in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_restarts, bounds.shape[0])):

            res = minimize(self.__objective__,
                           x0=start,
                           method="L-BFGS-B",
                           jac=self.__objective_grad__,
                           bounds=bounds,
                           args=fixed_params)

            if res.fun < min_objective:
                min_objective = res.fun
                best_theta = res.x

        best_theta = np.insert(best_theta, fixed_params, self.theta[fixed_params])

        return best_theta, -1 * min_objective

    def plot(self, show_data=True, bounds=None, fig=None):
        """ plot
        Plot the GP model's mean function and variance.
        Arguments:
        ----------
            show_data: boolean.
                Whether or not to display the training inputs X in the plot.
        """
        if self.X.shape[1] != 1:
            raise ValueError('Cannot plot with multi-dimensional inputs')
        if bounds is None:
            minVal = np.min(self.X)
            maxVal = np.max(self.X)
            delta = (maxVal - minVal) * self.__plot_delta__
            bounds = [np.floor(minVal - delta), np.ceil(maxVal + delta)]

        mean_x = np.linspace(bounds[0],
                             bounds[1],
                             self.__plot_density__).reshape(self.__plot_density__, self.kernel.dim)

        mean_y, cov = self.predict(mean_x)
        lq, uq = self.predict_quantiles(mean_x, noise=True)

        mean_x = mean_x.reshape(self.__plot_density__)
        if fig is None:
            fig = plt.figure()

        plt.plot(mean_x, mean_y, 'k', label='Predicted Mean')

        if show_data:
            plt.plot(self.X, self.Y, 'rx', label='Observed Inputs')

        plt.fill_between(mean_x, lq.reshape(lq.shape[0],),
                         uq.reshape(lq.shape[0],),
                         color='#B7E9F9', label='Confidence Bound')
        plt.ylim(bounds)
        plt.legend()

        return fig

