# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-08-04 22:36:14
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-08 15:14:28

import numpy as np


def parse_feature_vectors(X, features):
    """ parse_feature_vectors
    Parse an array of feature vectors (inputs) into an array of ValueChart Alternatives.
    ----------
        X: array-like, shape = [n_samples, n_features]
            The array of feature vectors to parse into Alternatives.
        features: array-like, shape = [n_features, ]
            The array of feature objects that map features to ValueChart Objectives.
    """
    alternatives = []

    for x in X:
        alternatives.append(parse_feature_vector(x, features))

    return alternatives


def parse_feature_vector(x, features):
    """
    parse_feature_vector
    Parse a single feature vector (input) into a ValueChart Alternative.
    ----------
        x: array-like, shape = [n_features, ]
            The array of feature vectors to parse into Alternatives.
        feature: array-like, shape = [n_features, ]
            The array of feature objects that map features to ValueChart Objectives.
    """

    name = str(int(np.random.rand(1)[0] * 1000))
    # How do i decide on these? Random generation?
    alternative = {'name': ('sample-' + name), 'id': ('temp-' + name), 'description': 'none'}
    objectiveValues = []

    for i, feature in enumerate(features):
        if feature['type'] == 'continuous':
            objectiveValues.append([feature['objective_name'], x[i]])
        elif x[i] == 1:
            objectiveValues.append([feature['objective_name'], feature['element']])

    alternative['objectiveValues'] = objectiveValues

    return alternative
