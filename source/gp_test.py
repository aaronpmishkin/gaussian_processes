# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-28 21:31:56
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-09 21:28:52

import numpy as np
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gaussian_process
import kernels
import mean_functions
import bayesian_optimization
from parse_value_chart import parse_valuechart

# Open and parse the ValueChart:
with open('../data/ValueCharts/Demonstration.json', 'r') as TestChart:
            value_chart = TestChart.read().replace('\n', '')

(value_chart,
 features,
 objective_map,
 bounds,
 X,
 utility_functions) = parse_valuechart(value_chart)

bounds = np.array(bounds)
bounds[:, 0] = 0
bounds[:, 1] = 1

# These are pulled from the ValueChart prior.
Y = np.array([[-0.1], [1.2]])

# Create the GP model:
rbf_kernel = kernels.RBF(dim=1, length_scale=1., var=1.)
additive_mean = mean_functions.AdditiveMean(dim=1, mean_functions=utility_functions)
gp = gaussian_process.GaussianProcess(X, Y, rbf_kernel, mean_function=additive_mean, obs_variance=0.005)

# Set the bounds for the hyperparameter optimization
opt_bounds = np.array([[1e-8, 1000], [1e-8, 1000], [1e-8, 1000]])

# Optimize the model and set the new hyperparameters:
theta, likelihood = gp.optimize(bounds=opt_bounds, fixed_params=[0])
print(theta, likelihood)
gp.set_hyperparameters(theta)

# Plot the GP before sampling:
gp.plot(legend=False)
plt.savefig(filename=("../figs/VC_init"))


x_next = bayesian_optimization.choose_sample(bayesian_optimization.upper_confidence_bound,
                                             gp,
                                             bounds=bounds)
print(x_next)

lengths = np.linspace(1e-8, 1, 500)
likelihoods = []


for length in lengths:
    likelihoods.append(gp.log_likelihood(np.array([0.05, length, 0.26])))

plt.figure()
plt.plot(lengths, likelihoods)
plt.savefig('../figs/likelihood_plot')




plt.show(block=True)





