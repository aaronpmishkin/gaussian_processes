# -*- coding: utf-8 -*-
# @Author: aaronpmishkin
# @Date:   2017-08-04 22:36:14
# @Last Modified by:   aaronpmishkin
# @Last Modified time: 2017-08-07 14:35:19

import unittest
import parse_value_chart as pvc
import parse_feature_vector as pfv
import json


class TestParseFeatureVector(unittest.TestCase):

    def setUp(self):
        with open('../data/ValueCharts/HotelChart.json', 'r') as HotelChart:
            self.data = HotelChart.read().replace('\n', '')
            self.value_chart = json.loads(self.data)
            _, features, objective_map, X, _, _ = pvc.parse_valuechart(self.data)
            self.features = features
            self.objective_map = objective_map
            self.X = X

    def test_parse_feature_vectors(self):
        alternatives = pfv.parse_feature_vectors(self.X, self.features)
        return


if __name__ == '__main__':
    unittest.main()
