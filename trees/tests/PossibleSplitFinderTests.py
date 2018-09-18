import unittest
import pandas as pd
from trees.PossibleSplitFinder import PossibleSplitFinder

class PossibleSplitFinderTests(unittest.TestCase):

    def setUp(self):
        self.label_column_name = 'label'
        self.possible_split_finder = PossibleSplitFinder()

    def tearDown(self):
        pass

    def test__find__no_records__returns_empty(self):
        feature = 'feature'
        empty_records = pd.DataFrame(columns=[feature])
        empty_labels = pd.DataFrame(columns=[self.label_column_name])

        subset = self.possible_split_finder.find(empty_records, empty_labels, feature)

        self.assertListEqual(subset, [])

    def test__find__single_record__returns_empty(self):
        feature = 'feature'
        single_record = pd.DataFrame([(5)], columns=[feature])
        single_label = pd.DataFrame([(0)], columns=[self.label_column_name])

        subset = self.possible_split_finder.find(single_record, single_label, feature)

        self.assertListEqual(subset, [])

    def test__find__multiple_records__returns_all_midpoints(self):
        feature1 = 'feature1'
        feature2 = 'feature2'
        records = pd.DataFrame([
            (5, 9),
            (10, 8),
            (13, 7),
            (3.5, 6)
        ],
            columns=[feature1, feature2])
        labels = pd.DataFrame([
            (0),
            (0),
            (1),
            (1),
        ],
            columns=[self.label_column_name])
        expected_possible_splits = [4.25, 11.5]

        subset = self.possible_split_finder.find(records, labels, feature1)

        self.assertListEqual(subset, expected_possible_splits)
