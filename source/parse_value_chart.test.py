# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-07-21 13:24:13
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 19:36:41


import numpy as np

import unittest
import parse_value_chart
import json


class test_parse_value_chart(unittest.TestCase):

    def setUp(self):
        with open('../data/ValueCharts/HotelChart.json', 'r') as HotelChart:
            self.data = HotelChart.read().replace('\n', '')
            self.valueChart = json.loads(self.data)

    def test_flatten_objectives(self):
        objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        self.assertEqual(len(objectives), 5)

    def test_parse_objectives(self):
        primitive_objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        features, objective_map = parse_value_chart.parse_objectives(primitive_objectives)
        self.assertEqual(len(features), 9)

    def test_parse_alternatives(self):
        primitive_objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        features, objective_map = parse_value_chart.parse_objectives(primitive_objectives)
        X = parse_value_chart.parse_alternatives(self.valueChart['alternatives'],
                                               features,
                                               objective_map)

        self.assertEqual(X.shape, (6, len(features)))

    def test_parse_discrete_domain_constraints(self):
        objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        constraints, index = parse_value_chart.parse_domain_constraints(objectives[0]['domain'], 3, 0)

        self.assertEqual(index, 3)
        self.assertEqual(len(constraints), 3)

        x = np.array([1, 2, 1])

        # Check the Constraint Functions:
        self.assertEqual(constraints[0]['fun'](x)[0], 3)
        self.assertEqual(constraints[1]['fun'](x)[0], 5)
        self.assertEqual(constraints[2]['fun'](x)[0], 9)

        # Check the Constraint Jacobians
        self.assertTrue(np.all(constraints[0]['jac'](x) == np.array([1, 1, 1])))
        self.assertTrue(np.all(constraints[1]['jac'](x) == np.array([2, 4, 2])))
        self.assertTrue(np.all(constraints[2]['jac'](x) == np.array([3, 12, 3])))

    def test_parse_continuous_domain_constraints(self):
        objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        constraints, index = parse_value_chart.parse_domain_constraints(objectives[1]['domain'], 1, 0)

        self.assertEqual(index, 1)
        self.assertEqual(len(constraints), 2)

        x = np.array([2])

        # Check the Constraint Functions:
        self.assertEqual(constraints[0]['fun'](x)[0], 1)
        self.assertEqual(constraints[1]['fun'](x)[0], 7)

        # Check the Constraint Jacobians
        self.assertEqual(constraints[0]['jac'](x)[0], 1)
        self.assertEqual(constraints[1]['jac'](x)[0], -1)

    def test_build_utility_functions(self):
        primitive_objectives = parse_value_chart.flatten_objectives(self.valueChart['rootObjectives'])
        features, objective_map, bounds = parse_value_chart.parse_objectives(primitive_objectives)

        utility_functions = parse_value_chart.build_utility_functions(self.valueChart['users'][0],
                                                                    features, objective_map)

        self.assertEqual(len(utility_functions), 5)

    def test_linear_interpolation(self):
        y = parse_value_chart.linear_interpolation([0, 2], [5, 6], 2.5)
        self.assertEqual(y, 4)

        y = parse_value_chart.linear_interpolation([5, 6], [8, 3], 5)
        self.assertEqual(y, 6)

    def test_parse_valuechart(self):
        (valuechart,
         features,
         objective_map,
         bounds,
         alternatives,
         utility_functions) = parse_value_chart.parse_valuechart(self.data)

        self.assertEqual(len(bounds), 9)
        self.assertEqual(len(features), 9)
        self.assertEqual(alternatives.shape, (6, len(features)))


if __name__ == '__main__':
    unittest.main()
