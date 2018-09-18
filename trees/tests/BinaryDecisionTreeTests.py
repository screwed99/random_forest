import unittest
from unittest.mock import patch
import pandas as pd
from trees.BinaryDecisionTree import BinaryDecisionTree

class BinaryDecisionTreeTests(unittest.TestCase):

    @patch('trees.FeatureSubsetter.FeatureSubsetter', autospec=True)
    @patch('trees.BinaryDecisionNode.BinaryDecisionNode', autospec=True)
    def setUp(self, start_root, feature_subsetter):
        self.features = ['feature1']
        self.root = start_root
        self.feature_subsetter = feature_subsetter
        self.binary_decision_tree = BinaryDecisionTree(self.features, self.root, self.feature_subsetter)

    def tearDown(self):
        pass

    def test__predict_label__root_is_leaf__returns_from_root(self):
        record = pd.DataFrame([(5)], columns=[self.features])
        mock_is_leaf = unittest.mock.Mock(return_value=True)
        self.root.is_leaf = mock_is_leaf
        expected_label = 1
        mock_predict_label = unittest.mock.Mock(return_value=expected_label)
        self.root.predict_label = mock_predict_label

        label = self.binary_decision_tree.predict_label(record)

        mock_is_leaf.assert_called_once()
        mock_predict_label.assert_called_once()
        self.assertEqual(label, expected_label)

    def test__predict_label__must_traverse__traverses_and_returns(self):
        record = pd.DataFrame([(5)], columns=[self.features])
        node1 = self.root
        mock_is_leaf1 = unittest.mock.Mock(return_value=False)
        node1.is_leaf = mock_is_leaf1
        node2 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode', autospec=True)
        mock_traverse_next_node1 = unittest.mock.Mock(return_value=node2)
        node1.traverse_next_node = mock_traverse_next_node1
        mock_is_leaf2 = unittest.mock.Mock(return_value=False)
        node2.is_leaf = mock_is_leaf2
        node3 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode', autospec=True)
        mock_traverse_next_node2 = unittest.mock.Mock(return_value=node3)
        node2.traverse_next_node = mock_traverse_next_node2
        mock_is_leaf3 = unittest.mock.Mock(return_value=True)
        node3.is_leaf = mock_is_leaf3
        expected_label = 1
        mock_predict_label = unittest.mock.Mock(return_value=expected_label)
        node3.predict_label = mock_predict_label

        label = self.binary_decision_tree.predict_label(record)

        mock_is_leaf1.assert_called_once()
        mock_traverse_next_node1.assert_called_once()
        mock_is_leaf2.assert_called_once()
        mock_traverse_next_node2.assert_called_once()
        mock_is_leaf3.assert_called_once()
        mock_predict_label.assert_called_once()
        self.assertEqual(label, expected_label)

    def test__try_add_best_split__root_only_leaf_with_no_split__returns_false(self):
        pass

    def test__try_add_best_split__many_leafs_and_splits__returns_best_split(self):
        pass