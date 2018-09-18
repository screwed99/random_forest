class BinaryDecisionTree:

    def __init__(self, features, start_root, feature_subsetter):
        self._features = features
        self._root = start_root
        self._max_height = 1
        self._used_features = set()
        self._curr_leaf_nodes = {[self._root]}
        self._feature_subsetter = feature_subsetter

    def predict_label(self, record):
        curr_node = self._root
        while not curr_node.is_leaf():
            curr_node = curr_node.traverse_next_node(record)
        return curr_node.predict_label()

    def try_add_best_split(self):
        allowed_features = self._feature_subsetter.subset(self._features, self._used_features)
        best_node_after_split = None
        best_node_before_split = None
        found_split = False
        for leaf_node in self._curr_leaf_nodes:
            node_after_split = leaf_node.get_best_split(allowed_features)
            if node_after_split is not None:
                if not found_split or node_after_split.best_binary_decision_node_split.compare(
                        best_node_after_split.binary_decision_node_split) == -1:
                    best_node_after_split = node_after_split
                    best_node_before_split = leaf_node
                    found_split = True
        if found_split:
            if best_node_before_split.parent is None:
                self._root = best_node_after_split
            else:
                if best_node_before_split.is_left_child:
                    best_node_before_split.parent.left_child = best_node_after_split
                else:
                    best_node_before_split.parent.right_child = best_node_after_split
            self._used_features.add(best_node_after_split.split_feature)
            self._curr_leaf_nodes.remove(best_node_before_split)
            self._curr_leaf_nodes.add(best_node_after_split.left_child)
            self._curr_leaf_nodes.add(best_node_after_split.right_child)
            if best_node_after_split.left_child.height > self._max_height:
                self._max_height = best_node_after_split.left_child.height
            return True
        return False


class BinaryDecisionTreeFactory:

    def __init__(self, feature_subsetter, binary_decision_node_factory):
        self._feature_subsetter = feature_subsetter
        self._binary_decision_node_factory = binary_decision_node_factory

    def create(self, records, labels):
        features = records.columns
        start_root = self._binary_decision_node_factory.create(records, labels, 1, None, None)
        return BinaryDecisionTree(features, start_root, self._feature_subsetter)
