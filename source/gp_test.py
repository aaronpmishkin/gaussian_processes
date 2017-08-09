# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-08 22:24:07

import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gaussian_process
import kernels
import bayesian_optimization

X = np.random.uniform(low=-3, high=3, size=(2, 1))

Y = np.sin(X) + np.random.normal(loc=0, scale=0.2, size=(2, 1))

print(X, Y)

rbf_kernel = kernels.RBF(dim=1, length_scale=2., var=1.)
gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel)


likelihood = gp.log_likelihood()
grad_likelihood = gp.log_likelihood_gradient()
# print('My Likelihood: ', likelihood, grad_likelihood)

bounds = np.array([[0.001, 10], [0.001, 10], [0.001, 10]])

# theta, op_likelihood = gp.optimize(bounds=bounds, n_restarts=20)
theta = np.array([0.002, 1.1, 0.3])
gp.set_hyperparameters(theta)
# print('Optimized RBF Model: ', theta, op_likelihood)

gp.plot(bounds=[-3, 3])

sin_x = np.linspace(-3, 3, 100)
sin_y = np.sin(sin_x)

plt.plot(sin_x, sin_y, 'g', label='Ground Truth')
plt.legend()

plt.savefig(filename=("../figs/initial_samples"))

for i in range(5):
    x_next = bayesian_optimization.choose_sample(bayesian_optimization.upper_confidence_bound,
                                                 gp,
                                                 bounds=np.array([[-3, 3]]))
    y_next = np.sin(x_next) + np.random.normal(loc=0, scale=0.1, size=(1,))

    print(x_next, y_next)
    X = np.append(X, x_next, axis=0)
    Y = np.append(Y, y_next, axis=0)

    gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel)
    gp.set_hyperparameters(theta)

    gp.plot(bounds=[-3, 3])
    plt.plot(sin_x, sin_y, 'g', label='Ground Truth')
    plt.legend()

    plt.savefig(filename=("../figs/sample_" + str(i)))

plt.show(block=True)
