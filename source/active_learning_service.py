# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-08-07 12:34:15
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 20:35:08

# Import Libraries
import numpy as np
from nameko.extensions import DependencyProvider
from nameko.web.websocket import WebSocketHubProvider, rpc

# Import Application Classes:
from parse_value_chart import parse_valuechart
from parse_feature_vector import parse_feature_vectors
import bayesian_optimization
from gaussian_process import GaussianProcess
from kernels import RBF, Additive
from mean_functions import AdditiveMean


class Config(DependencyProvider):
    def get_dependency(self, worker_ctx):
        return self.container.config


class ContainerIdentifier(DependencyProvider):
    def get_dependency(self, worker_ctx):
        return id(self.container)


class ActiveLearningService(object):

    name = 'active-learning-service'

    websocket_hub = WebSocketHubProvider()
    container_id = ContainerIdentifier()
    config = Config()

    @rpc
    def next_alternative(self, socket_id, json_chart, scores):
        (valuechart,
         features,
         objective_map,
         bounds,
         X,
         utility_functions) = parse_valuechart(json_chart)

        bounds = np.array(bounds)
        Y = np.array(scores).reshape(len(scores), 1)

        base_kernels = []
        for feature in features:
            base_kernels.append(RBF(dim=1))
        base_kernels = np.array(base_kernels)

        additive_kernel = Additive(dim=len(features), order=1, base_kernels=base_kernels)
        additive_mean = AdditiveMean(dim=len(features), mean_functions=utility_functions)

        gp = GaussianProcess(X, Y, additive_kernel, mean_function=additive_mean)
        theta, likelihood = gp.optimize()
        gp.set_hyperparameters(theta)

        x_star = bayesian_optimization.choose_sample(bayesian_optimization.upper_confidence_bound,
                                                     gp,
                                                     n_features=len(features),
                                                     bounds=bounds)
        x_star = bayesian_optimization.choose_categories(
            x_star,
            bayesian_optimization.upper_confidence_bound,
            gp, objective_map)

        alternative = parse_feature_vectors(x_star, features)
        print(x_star)

        return {'alternative': alternative}
