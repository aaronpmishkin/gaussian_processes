# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-10 14:41:16
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-04 16:17:18

import numpy as np
from scipy.optimize import minimize


def upper_confidence_bound(x, gp, quantiles=(2.5, 97.5), maximization=True):
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

    confidence_bounds = gp.predict_quantiles(np.array([x]), quantiles)

    if (maximization):
        ucb = confidence_bounds[1][0]
    else:
        ucb = confidence_bounds[0][0]

    scaling_factor = (-1) ** (maximization)

    return scaling_factor * ucb


def choose_sample(acquisition_function, gp, n_features, bounds, constraints, n_restarts=10, maximization=True):
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
    best_acquisition_value = 1

    for start in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_features)):

        res = minimize(acquisition_function,
                       x0=start,
                       constraints=constraints,
                       method="SLSQP",
                       args=(gp, (2.5, 97.5), maximization))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return np.array([best_x])


def update_process(gp, X, Y, Xnew, Ynew):
    """ update_process

    Update the given gaussian process model to include new function evaluations

    Arguments:
    ----------
        gp: GPy.models.GP
            Gaussian process trained on previous function evaluations.
        X: array-like, shape = [n_samples, n_features]
            The point(s) for which the expected improvement needs to be
            computed.
        Y: array-like, shape = [n_samples, n_ouputs]
            The point(s) for which the expected improvement needs to be
            computed.
        Xnew: array-like, shape = [n_samples, n_features]
            The point(s) for which the expected improvement needs to be
            computed.
        Ynew: array-like, shape = [n_samples, n_outputs]
            The point(s) for which the expected improvement needs to be
            computed.
    """

    X = np.append(X, Xnew, axis=0)
    Y = np.append(Y, Ynew, axis=0)
    gp.set_XY(X, Y)
    gp.optimize()

    return gp, X, Y
