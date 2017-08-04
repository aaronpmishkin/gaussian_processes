# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-04 14:03:28

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import gaussian_process
import kernels
import mean_functions

X = np.random.uniform(low=-3, high=3, size=5)
X = X.reshape(5, 1)
Y = np.sin(X) + np.random.normal(loc=0, scale=0.1, size=X.shape)

sin_x = np.linspace(-3, 3, 100)
sin_y = np.sin(sin_x)

base_kernel = kernels.RBF(dim=1, length_scale=1, var=1)
additive_kernel = kernels.Additive(dim=1, order=1, base_kernels=[base_kernel])
gp_additive = gaussian_process.GaussianProcess(X, Y, additive_kernel, mean_functions.zero_mean, obs_variance=0.1)

likelihood = gp_additive.log_likelihood(np.array([0.1, 1, 1, 1]))
grad_likelihood = gp_additive.log_likelihood_gradient(np.array([0.1, 1, 1, 1]))

print('GP Additive Likelihood:', likelihood, grad_likelihood)

rbf_kernel = kernels.RBF(dim=1, length_scale=1, var=1)
gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel, mean_functions.zero_mean, obs_variance=0.1)

likelihood = gp.log_likelihood(np.array([0.1, 1, 1]))
grad_likelihood = gp.log_likelihood_gradient(np.array([0.1, 1, 1]))


print('My Likelihood: ', likelihood, grad_likelihood)

import GPy

k = GPy.kern.RBF(1, lengthscale=1)
gp_GPy = GPy.models.GPRegression(X, Y, k, noise_var=0.1)
print('GPy Likelihood: ', gp_GPy.log_likelihood(), gp_GPy.objective_function_gradients())


theta, op_likelihood = gp_additive.optimize()
gp_additive.set_hyperparameters(theta)
print('Optimized Additive Model: ', theta, op_likelihood)


theta, op_likelihood = gp.optimize()
gp.set_hyperparameters(theta)
print('Optimized RBF Model: ', theta, op_likelihood)

gp_GPy.optimize()
print(gp_GPy)


gp_additive.plot()
gp.plot()
gp_GPy.plot()

plt.show()
