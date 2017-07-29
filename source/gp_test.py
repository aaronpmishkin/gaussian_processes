# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-07-28 23:00:25


import numpy as np
import matplotlib.pyplot as plt

import gaussian_process
import kernels
import mean_functions

X = np.random.uniform(low=-3, high=3, size=5)
X = X.reshape(5, 1)
Y = np.sin(X) + np.random.normal(loc=0, scale=0.1, size=X.shape)

sin_x = np.linspace(-3, 3, 500)
sin_y = np.sin(sin_x)

# Plot the ground truth:
plt.figure()
plt.plot(X, Y, 'x', sin_x, sin_y)
plt.legend(['Sampled Data', 'Ground Truth'])

rbf_kernel = kernels.RBF(length_scale=1, sigma=1)


gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel, mean_functions.zero_mean, sigma=0.1)

X_new = np.linspace(-3, 3, 10)
X_new = X_new.reshape(10, 1)
Y_new, cov = gp.predict(X_new, noise=True)
lq, uq = gp.predict_quantiles(X_new, noise=True)

likelihood = gp.log_likelihood()
print(likelihood)

plt.figure()
plt.plot(X_new.reshape(10,), Y_new, '*', X, Y, 'x', sin_x, sin_y, '-', X_new, lq, '_', X_new, uq, '_')
plt.legend(['Predictions', 'Initial Inputs', 'Ground Truth'])

plt.show()
