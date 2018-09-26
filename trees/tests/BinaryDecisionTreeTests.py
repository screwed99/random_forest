import unittest
from unittest.mock import patch
import pandas as pd
from copy import deepcopy

from trees.BinaryDecisionTree import BinaryDecisionTree


class CopyingMock(unittest.mock.MagicMock):
    def __call__(self, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super(CopyingMock, self).__call__(*args, **kwargs)

class BinaryDecisionTreeTests(unittest.TestCase):

    @patch('numpy.random.RandomState', autospec=True)
    @patch('trees.LeafNodes.LeafNodes', autospec=True)
    @patch('trees.NodeSplitter.NodeSplitter', autospec=True)
    @patch('trees.FeatureSubsetter.FeatureSubsetter', autospec=True)
    @patch('trees.BinaryDecisionNode.BinaryDecisionNode', autospec=True)
    def setUp(self, start_root, feature_subsetter, node_splitter, leaf_nodes, random_state):
        self.features = ['feature1']
        self.root = start_root
        self.feature_subsetter = CopyingMock(feature_subsetter)
        self.node_splitter = node_splitter
        self.leaf_nodes = leaf_nodes
        self.random_state = random_state
        self.binary_decision_tree = BinaryDecisionTree(
            self.features, self.root, self.feature_subsetter, self.node_splitter, self.leaf_nodes,
            self.random_state)

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

    def test__try_add_best_split__successful_splits__updates_root_and_height(self):
        split_feature1 = "split_feature1"
        split_feature2 = "split_feature2"
        allowed_features1 = [split_feature1, split_feature2]
        used_features1 = set()
        subset_call1 = unittest.mock.call(used_features1, self.random_state)
        self.feature_subsetter.subset.return_value = allowed_features1
        get_leaf_nodes_call1 = unittest.mock.call()
        mock_leaf_nodes1 = unittest.mock.Mock()
        self.leaf_nodes.get_leaf_nodes.return_value = mock_leaf_nodes1
        split_call1 = unittest.mock.call(mock_leaf_nodes1, allowed_features1)
        mock_node_split_result1 = unittest.mock.patch('trees.NodeSplitResult.NodeSplitResult',
                                                     autospec=True)
        best_node_before_split1 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode',
                                                     autospec=True)
        best_node_after_split1 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode',
                                                     autospec=True)
        mock_found_split1 = unittest.mock.PropertyMock(return_value=True)
        type(mock_node_split_result1).found_split = mock_found_split1
        type(mock_node_split_result1).best_node_before_split = best_node_before_split1
        type(mock_node_split_result1).best_node_after_split = best_node_after_split1
        type(best_node_before_split1).parent = None
        update_leaf_call1 = unittest.mock.call(best_node_before_split1, best_node_after_split1)
        mock_after_split_left_child1 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode',
                                                     autospec=True)
        type(best_node_after_split1).left_child = mock_after_split_left_child1
        type(best_node_after_split1).split_feature = split_feature1
        type(mock_after_split_left_child1).height = 2
        self.node_splitter.split.return_value = mock_node_split_result1

        split_succeeded = self.binary_decision_tree.try_add_best_split()

        self.feature_subsetter.subset.assert_has_calls([subset_call1])
        self.leaf_nodes.get_leaf_nodes.assert_has_calls([get_leaf_nodes_call1])
        self.node_splitter.split.assert_has_calls([split_call1])
        mock_leaf_nodes1.assert_not_called()
        mock_found_split1.assert_called_once()
        self.leaf_nodes.update_leaf_node_split.assert_has_calls([update_leaf_call1])
        self.assertTrue(split_succeeded)
        self.assertEqual(self.binary_decision_tree.get_max_height(), 2)


        """
        allowed_features2 = [split_feature2]
        used_features2 = {split_feature1}
        subset_call2 = unittest.mock.call(used_features2, self.random_state)
        self.feature_subsetter.subset.return_value = allowed_features2
        get_leaf_nodes_call2 = unittest.mock.call()
        mock_leaf_nodes2 = unittest.mock.Mock()
        self.leaf_nodes.get_leaf_nodes.return_value = mock_leaf_nodes2
        split_call2 = unittest.mock.call(mock_leaf_nodes2, allowed_features2)
        mock_node_split_result2 = unittest.mock.patch('trees.NodeSplitResult.NodeSplitResult',
                                                     autospec=True)
        best_node_before_split2 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode',
                                                     autospec=True)
        best_node_after_split2 = unittest.mock.patch('trees.BinaryDecisionNode.BinaryDecisionNode',
                                                     autospec=True)
        mock_node_split_result2.found_split.return_value = True
        mock_node_split_result2.best_node_before_split.return_value = best_node_before_split2
        mock_node_split_result2.best_node_after_split.return_value = best_node_after_split2
        best_node_before_split1.parent.return_value = parent_of_split_node2
        best_node_before_split1.is_left_child.return_value = True
                parent_of_split_node2.assert_has_calls()

        self.node_splitter.split.return_value = mock_node_split_result2

        split_succeeded = self.binary_decision_tree.try_add_best_split()

        self.feature_subsetter.subset.assert_has_calls([subset_call2])
        self.leaf_nodes.get_leaf_nodes.assert_has_calls([get_leaf_nodes_call2])
        self.node_splitter.split.assert_has_calls([split_call2])
        mock_leaf_nodes2.assert_not_called()
        mock_node_split_result2.found_split.assert_called_once()
        self.assertTrue(split_succeeded)
        self.assertEqual(self.binary_decision_tree.get_max_height(), 3)
        """

    def test__try_add_best_split__unsuccessful_split__does_nothing(self):
        allowed_features = ["allowed_feature"]
        used_features = set()
        subset_call = unittest.mock.call(used_features, self.random_state)
        self.feature_subsetter.subset.return_value = allowed_features
        get_leaf_nodes_call = unittest.mock.call()
        mock_leaf_nodes = unittest.mock.Mock()
        self.leaf_nodes.get_leaf_nodes.return_value = mock_leaf_nodes
        split_call = unittest.mock.call(mock_leaf_nodes, allowed_features)
        mock_node_split_result = unittest.mock.patch('trees.NodeSplitResult.NodeSplitResult',
                                                     autospec=True)
        mock_found_split = unittest.mock.PropertyMock(return_value=False)
        type(mock_node_split_result).found_split = mock_found_split
        self.node_splitter.split.return_value = mock_node_split_result

        split_succeeded = self.binary_decision_tree.try_add_best_split()

        self.feature_subsetter.subset.assert_has_calls([subset_call])
        self.leaf_nodes.get_leaf_nodes.assert_has_calls([get_leaf_nodes_call])
        self.node_splitter.split.assert_has_calls([split_call])
        mock_leaf_nodes.assert_not_called()
        mock_found_split.assert_called_once()
        self.assertFalse(split_succeeded)
        self.assertEqual(self.binary_decision_tree.get_max_height(), 1)