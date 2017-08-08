# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-10 14:41:16
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 20:28:54

import numpy as np
from scipy.optimize import minimize


def upper_confidence_bound(x, gp, quantiles=(0.025, 0.975), maximization=True):
    """ upper_confidence_bound
    Upper Confidence Bound (UCB) Acquisition Function
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_features]
            The point(s) for which the expected improvement needs to be
            computed.
        gp: GPy.models.GP
            Gaussian process trained on previous function evaluations.
        quantiles: int
            quantiles for the GP posterior distribution
            controls the size of the confidence bound
        maximization: boolean
            Boolean flag indicating whether or not we want the upper
            or lower confidence bound
    """

    lq, uq = gp.predict_quantiles(np.array([x]), confidence_bounds=quantiles)

    if (maximization):
        ucb = np.sum(uq)    # Hack to get the single value out of the nested array.
    else:
        ucb = np.sum(lq)

    scaling_factor = (-1) ** (maximization)

    return scaling_factor * ucb


def choose_sample(acquisition_function, gp, n_features, bounds, n_restarts=10, maximization=True):
    """ choose_sample
    Choose the next point to sample based on the provided acquisition_function
    gp model, and constraints.
    Arguments:
    ----------
        acquisition_func: function
            The acquisition function to optimize.
        gp: GPy.models.GP
            Gaussian process trained on previously evaluated points.
        n_features: int
            The number of features for each object in X
        bounds: array-like, shape = [n_features, 2]
            The upper and lower bounds for each feature in X
        constraints: sequence
            Sequence of dictionaries specifying constraints on valid examples X
        n_restarts:
            Number of restarts for the optimizer
        maximization: boolean.
            Boolean flag that indicates whether the loss function
            is to be maximized or minimized.
    """

    best_x = None
    best_acquisition_value = np.Infinity

    for start in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_features)):

        res = minimize(acquisition_function,
                       x0=start,
                       bounds=bounds,
                       method="SLSQP",
                       args=(gp))

        if res.fun < best_acquisition_value and res.success:
            best_acquisition_value = res.fun
            best_x = res.x

    return np.array([best_x])


def choose_category(x, acquisition_function, gp, features):
    for key, feature in features.items():
        x[feature['index']] = 0

    best_acquisition_value = np.Infinity

    for key, feature in features.items():
        x_hat = np.copy(x)
        x_hat[feature['index']] = 1
        acq_value = acquisition_function(x_hat, gp)

        if acq_value < best_acquisition_value:
            best_x = x_hat
            best_acquisition_value = acq_value

    return best_x


def choose_categories(x, acquisition_function, gp, objective_map):
    x = x[0]
    for key, objective in objective_map.items():
        if objective['type'] == 'discrete':
            x = choose_category(x, acquisition_function, gp, objective['elements'])

    return np.array([x])


