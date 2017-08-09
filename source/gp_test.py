# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-08 19:55:36

import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import GPy
import gaussian_process
import kernels
import bayesian_optimization

X = np.random.uniform(low=-5, high=5, size=(5, 1))

Y = -X

print(X, Y)

rbf_kernel = kernels.RBF(dim=1, length_scale=1., var=1.)
gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel, obs_variance=1.)

gp.plot()

likelihood = gp.log_likelihood(np.array([1., 10.68, 1.]))
grad_likelihood = gp.log_likelihood_gradient(np.array([1., 10.68, 1.]))
print('My Likelihood: ', likelihood, grad_likelihood)

# Double Check that the Likelihood and Gradients are correct:
k = GPy.kern.RBF(1, lengthscale=10.68)
gp_GPy = GPy.models.GPRegression(X, Y, k, noise_var=1)
print('GPy Likelihood: ', gp_GPy.log_likelihood(), gp_GPy.objective_function_gradients())

gp_GPy.plot()

theta, op_likelihood = gp.optimize()
gp.set_hyperparameters(theta)
print('Optimized RBF Model: ', theta, op_likelihood)

gp.plot()

gp_GPy.optimize()
print(gp_GPy)
gp_GPy.plot()
plt.show()

