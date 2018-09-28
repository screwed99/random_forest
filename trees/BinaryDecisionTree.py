from trees.LeafNodes import LeafNodes


class BinaryDecisionTree:

    def __init__(self, features, start_root, feature_subsetter, node_splitter, leaf_nodes,
                 random_state):
        self._features = features
        self._root = start_root
        self._max_height = 1
        self._used_features = set()
        self._curr_leaf_nodes = {self._root}
        self._feature_subsetter = feature_subsetter
        self._node_splitter = node_splitter
        self._leaf_nodes = leaf_nodes
        self._random_state = random_state

    def predict_label(self, record):
        curr_node = self._root
        while not curr_node.is_leaf():
            curr_node = curr_node.traverse_next_node(record)
        return curr_node.predict_label()

    def try_add_best_split(self):
        allowed_features = self._feature_subsetter.subset(self._used_features, self._random_state)
        leaf_nodes = self._leaf_nodes.get_leaf_nodes()
        node_split_result = self._node_splitter.split(leaf_nodes, allowed_features)
        if node_split_result.found_split:
            self._resolve_successful_split(node_split_result)
            return True
        return False

    def get_max_height(self):
        return self._max_height

    def _resolve_successful_split(self, node_split_result):
        best_node_before_split = node_split_result.best_node_before_split
        best_node_after_split = node_split_result.best_node_after_split
        if best_node_before_split.parent is None:
            self._root = best_node_after_split
        else:
            if best_node_before_split.is_left_child:
                best_node_before_split.parent.left_child = best_node_after_split
            else:
                best_node_before_split.parent.right_child = best_node_after_split
        self._used_features.add(best_node_after_split.split_feature)
        self._leaf_nodes.update_leaf_node_split(best_node_before_split=best_node_before_split,
                                                best_node_after_split=best_node_after_split)
        if best_node_after_split.height > self._max_height:
            self._max_height = best_node_after_split.height


class BinaryDecisionTreeFactory:

    def __init__(self, feature_subsetter, binary_decision_node_factory, node_splitter):
        self._feature_subsetter = feature_subsetter
        self._binary_decision_node_factory = binary_decision_node_factory
        self._node_splitter = node_splitter

    def create(self, records, labels, random_state):
        features = records.columns
        start_root = self._binary_decision_node_factory.create(records, labels, 1, None, None)
        leaf_nodes = LeafNodes([start_root])
        return BinaryDecisionTree(features, start_root, self._feature_subsetter,
                                  self._node_splitter, leaf_nodes, random_state)
