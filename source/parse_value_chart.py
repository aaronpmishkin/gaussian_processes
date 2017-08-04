# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-11 11:20:08
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-04 16:02:55

import numpy as np
import json


def parse_valuechart(value_chart_json):
    value_chart = json.loads(value_chart_json)

    primitive_objectives = flatten_objectives(value_chart['rootObjectives'])
    features, objective_map = parse_objectives(primitive_objectives)
    alternatives = parse_alternatives(value_chart['alternatives'], features, objective_map)
    utility_functions = build_utility_functions(value_chart['users'][0], features, objective_map)
    constraints = build_constraints(primitive_objectives, len(features))

    return features, objective_map, alternatives, utility_functions, constraints


def flatten_objectives(objectives):
    primitive_objectives = []

    for objective in objectives:
        if objective['objectiveType'] == 'primitive':
            primitive_objectives.append(objective)
        else:
            primitive_objectives.extend(
                flatten_objectives(objective['subObjectives']))

    return primitive_objectives


def parse_objectives(primitive_objectives):
    features = []
    objective_map = {}
    feature_index = 0

    for index, objective in enumerate(primitive_objectives):

        if objective['domain']['type'] == 'continuous':
            feature = {'objective_name': objective['name']}
            feature['type'] = 'continuous'
            feature['bounds'] = (objective['domain']['minValue'], objective['domain']['maxValue'])
            feature['index'] = feature_index

            objective_map[objective['name']] = feature

            features.append(feature)
            feature_index += 1
        else:
            elements = {}
            for element in objective['domain']['elements']:
                feature = {'objective_name': objective['name']}
                feature['type'] = 'discrete'
                feature['bounds'] = (0, 1)
                feature['element'] = element
                feature['index'] = feature_index

                features.append(feature)
                elements[element] = feature
                feature_index += 1

            objective_map[objective['name']] = {'elements': elements, 'type': 'discrete'}

    return features, objective_map


def parse_alternatives(alternatives, features, objective_map):
    X = np.array([])

    for alternative in alternatives:
        xnew = parse_alternative(alternative, features, objective_map)
        X = np.append(X, xnew, axis=0)

    return np.reshape(X, (len(alternatives), len(features)))


def parse_alternative(alternative, features, objective_map):
    x = np.zeros(len(features))

    for outcome in alternative['objectiveValues']:
        feature = objective_map[outcome[0]]

        if feature['type'] == 'continuous':
            x[feature['index']] = outcome[1]
        else:
            for key in feature['elements']:
                element = feature['elements'][key]
                if (element['element'] == outcome[1]):
                    x[element['index']] = 1
                else:
                    x[element['index']] = 0

    return x


def build_utility_functions(user, features, objective_map):
    u_functions = {}
    weights = user['weightMap']['weights']
    score_functions = user['scoreFunctionMap']['scoreFunctions']

    for score_function in score_functions:
        w = get_weight(weights, score_function[0])

        if (score_function[1]['type'] == 'continuous'):
            u_functions[score_function[0]] = build_continuous_utility(
                score_function[1],
                w,
                features,
                objective_map[score_function[0]])
        else:
            u_functions[score_function[0]] = build_discrete_utility(
                score_function[1],
                w,
                features,
                objective_map[score_function[0]]['elements'])

    return u_functions


def build_continuous_utility(score_function, weight, features, continuous_feature):
    index = continuous_feature['index']
    scores = np.array(score_function['elementScoreMap'])

    def f(x):
        start = scores[scores[:, 0] <= x[index]]
        end = scores[scores[:, 0] >= x[index]]

        if len(start) != 0:
            start = start[-1]
        else:
            start = scores[0]

        if len(end) != 0:
            end = end[0]
        else:
            end = scores[-1]

        if start[0] == end[0]:
            return weight * start[1]
        else:
            return weight * linear_interpolation(start, end, x[index])

    def df(x):
        start = scores[scores[:, 0] <= x[index]]
        end = scores[scores[:, 0] >= x[index]]

        if len(start) != 0:
            start = start[-1]
        else:
            start = scores[0]

        if len(end) != 0:
            end = end[0]
        else:
            end = scores[-1]

        if start[0] == end[0]:
            return 0            # The gradient is not defined at this point.
        else:
            return weight * (end[1] - start[1]) / (end[0] - start[0])

    return f, df


def build_discrete_utility(score_function, weight, features, discrete_features):

    scores = np.zeros(len(features))

    for element in score_function['elementScoreMap']:
        scores[discrete_features[element[0]]['index']] = element[1]

    def f(x):
        return weight * np.dot(x, scores)

    def df(x):
        return weight * scores

    return f, df


def linear_interpolation(start, end, element):
    slope = (end[1] - start[1]) / (end[0] - start[0])
    offset = start[1] - (slope * start[0])

    return (slope * element) + offset


def get_weight(weights, objective_name):
    return list(filter(lambda x: x[0] == objective_name, weights))[0][1]


def build_constraints(primitive_objectives, num_features):
    constraints = []

    i = 0
    for objective in primitive_objectives:
        c, i = parse_domain_constraints(objective['domain'], num_features, i)
        constraints.extend(c)

    return constraints


def parse_domain_constraints(domain, num_features, i):
    constraints = []

    if domain['type'] == 'continuous':
        min_value = domain['minValue']
        max_value = domain['maxValue']

        jac = np.zeros(num_features)
        jac[i] = 1

        constraints.append({
            'type': 'ineq',
            'fun': build_continuous_fun(i, min_value, max=False),
            'jac': build_continuous_jac(i, num_features, max=False)
        })

        constraints.append({
            'type': 'ineq',
            'fun': build_continuous_fun(i, max_value, max=True),
            'jac': build_continuous_jac(i, num_features, max=True)
        })

        i += 1

    else:
        elements = domain['elements']

        for j, element in enumerate(elements):
            constraints.append({
                'type': 'eq',
                'fun': build_categorical_fun(num_features, i, j, elements),
                'jac': build_categorical_jac(num_features, i, j, elements)
            })

        i += len(elements)

    return constraints, i


# Functions for creating Functions and Jacobians for Constraints


def build_continuous_fun(i, value, max=False):
    if (max):
        def f(x):
            return np.array([value - x[i]])
    else:
        def f(x):
            return np.array([x[i] - value])

    return f


def build_continuous_jac(i, num_features, max=False):
    def f(x):
        jac = np.zeros(num_features)
        jac[i] = 1

        if max:
            jac = jac * -1

        return jac

    return f


def build_categorical_fun(num_features, i, j, elements):
    def f(x):
        return np.array([np.sum(x[i: i + len(elements)] ** (j + 1))]) - 1

    return f


def build_categorical_jac(num_features, i, j, elements):
    def f(x):
        jac = np.zeros(num_features)
        jac[i:i + len(elements)] = x[i: i + len(elements)]
        jac = (j + 1) * (jac ** j)
        return jac

    return f
