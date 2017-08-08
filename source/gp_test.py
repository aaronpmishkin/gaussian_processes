# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 20:30:38

import numpy as np
import matplotlib.pyplot as plt
import GPy
import gaussian_process
import kernels
import bayesian_optimization
import parse_value_chart
from mean_functions import AdditiveMean


# X = np.random.uniform(low=[0, 0], high=[10, 10], size=(5, 2))


# Y = (X[:, 0] ** 2) + X[:, 1] + np.random.normal(loc=0, scale=4, size=(5,))
# Y = Y.reshape(5,1)

# print(X, Y)
# base_kernel_0 = kernels.RBF(dim=1, length_scale=1, var=1)
# base_kernel_1 = kernels.RBF(dim=1, length_scale=1, var=1)

# additive_kernel = kernels.Additive(dim=2, order=2, base_kernels=[base_kernel_0, base_kernel_1])
# gp_additive = gaussian_process.GaussianProcess(X, Y, additive_kernel, obs_variance=1)

# likelihood = gp_additive.log_likelihood(np.array([1, 1, 1, 1, 1, 1, 1]))
# grad_likelihood = gp_additive.log_likelihood_gradient(np.array([1, 1, 1, 1, 1, 1, 1]))

# print('GP Additive Likelihood:', likelihood, grad_likelihood)

# rbf_kernel = kernels.RBF(dim=1, length_scale=1, var=1)
# gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel, obs_variance=1)

# likelihood = gp.log_likelihood(np.array([1, 1, 1]))
# grad_likelihood = gp.log_likelihood_gradient(np.array([0.1, 1, 1]))


# print('My Likelihood: ', likelihood, grad_likelihood)


# k = GPy.kern.RBF(2, lengthscale=1)
# gp_GPy = GPy.models.GPRegression(X, Y, k, noise_var=1)
# print('GPy Likelihood: ', gp_GPy.log_likelihood(), gp_GPy.objective_function_gradients())


# theta, op_likelihood = gp_additive.optimize()
# print('Optimized Additive Model: ', theta, op_likelihood)
# gp_additive.set_hyperparameters(theta)


# theta, op_likelihood = gp.optimize()
# gp.set_hyperparameters(theta)
# print('Optimized RBF Model: ', theta, op_likelihood)

# gp_GPy.optimize()
# print(gp_GPy)


with open('../data/ValueCharts/HotelChart.json', 'r') as TestChart:
            value_chart = TestChart.read().replace('\n', '')

(value_chart,
 features,
 objective_map,
 bounds,
 X,
 utility_functions) = parse_value_chart.parse_valuechart(value_chart)

print(X)
Y = np.array([27, 75, 42, 53, 86, 61]).reshape(X.shape[0], 1)
additive_kernel = kernels.Additive(dim=5, order=1, base_kernels=[kernels.RBF(dim=1),
                                                                 kernels.RBF(dim=1),
                                                                 kernels.RBF(dim=1),
                                                                 kernels.RBF(dim=1),
                                                                 kernels.RBF(dim=1)])
additive_mean = AdditiveMean(dim=2, mean_functions=utility_functions)
gp = gaussian_process.GaussianProcess(X, Y, additive_kernel, mean_function=additive_mean)
theta, op_likelihood = gp.optimize()
gp.set_hyperparameters(theta)

bounds = np.array(bounds)
x = bayesian_optimization.choose_sample(bayesian_optimization.upper_confidence_bound,
                                        gp,
                                        X.shape[1],
                                        bounds,
                                        10,
                                        True)
print(x)

x = bayesian_optimization.choose_categories(
    x,
    bayesian_optimization.upper_confidence_bound,
    gp, objective_map)

print(x)
