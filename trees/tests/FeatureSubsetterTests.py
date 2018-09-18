import unittest
from unittest.mock import patch
from trees.FeatureSubsetter import FeatureSubsetter

class FeatureSubsetterTests(unittest.TestCase):

    def setUp(self):
        self.feature_subsetter = FeatureSubsetter()

    def tearDown(self):
        pass

    @patch('numpy.random.RandomState', autospec=True)
    def test__subset__no_unused_features__returns_empty(self, mock_random_state):
        features = set(range(10))
        used_features = set(range(10))

        subset = self.feature_subsetter.subset(features, used_features, mock_random_state)

        self.assertFalse(subset)
        mock_random_state.assert_not_called()

    @patch('numpy.random.RandomState', autospec=True)
    def test__subset__only_one_unused_feature__returns_single_feature(self, mock_random_state):
        mock_rand_int = unittest.mock.Mock(return_value=0)
        mock_random_state.rand_int = mock_rand_int
        features = set(range(10))
        used_features = set(range(9))
        unused_feature = 9

        subset = self.feature_subsetter.subset(features, used_features, mock_random_state)

        mock_rand_int.assert_called_once_with(1)
        self.assertListEqual(subset, [unused_feature])


    @patch('numpy.random.RandomState', autospec=True)
    def test__subset__multiple_unused_features__returns_square_root(self, mock_random_state):
        features = set(range(10))
        used_features = {0}
        rand_int_ranges = [unittest.mock.call(9), unittest.mock.call(8), unittest.mock.call(7)]
        rand_ints = [6, 3, 4]
        expected_subset = [7, 4, 6]
        mock_rand_int = unittest.mock.Mock(side_effect=rand_ints)
        mock_random_state.rand_int = mock_rand_int

        subset = self.feature_subsetter.subset(features, used_features, mock_random_state)

        self.assertListEqual(mock_rand_int.call_args_list, rand_int_ranges)
        self.assertListEqual(subset, expected_subset)
