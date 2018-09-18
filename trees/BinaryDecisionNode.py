class BinaryDecisionNode:

    def __init__(self, records, labels, height, parent, is_left_child, possible_split_finder,
                 best_split_finder, binary_decision_node_factory):
        self._records = records
        self._labels = labels
        self.height = height
        self.parent = parent
        self.is_left_child = is_left_child
        self._less_than_split_point = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.binary_decision_node_split = None
        self._feature_to_possible_splits = {}
        self._possible_split_finder = possible_split_finder
        self._best_split_finder = best_split_finder
        self._binary_decision_node_factory = binary_decision_node_factory

    @classmethod
    def with_split(cls, records, labels, height, parent, is_left_child, possible_split_finder,
                   best_split_finder, binary_decision_node_factory, binary_decision_node_split):
        inst = cls(records, labels, height, parent, is_left_child, possible_split_finder,
                   best_split_finder, binary_decision_node_factory)
        inst._less_than_split_point = binary_decision_node_split.split_point
        inst.left_child = binary_decision_node_split.child_left,
        inst.right_child = binary_decision_node_split.child_right,
        inst.split_feature = binary_decision_node_split.split_feature
        return inst

    def is_leaf(self):
        return self.left_child is None

    def get_best_split(self, allowed_features):
        # TODO consider different handling of equally good splits
        found_split = False
        best_binary_decision_node_split = None
        for feature in allowed_features:
            if feature in self._feature_to_possible_splits:
                possible_splits = self._feature_to_possible_splits[feature]
            else:
                possible_splits = self._possible_split_finder.find(self._records, self._labels,
                                                                   feature)
                self._feature_to_possible_splits[feature] = possible_splits
            binary_decision_node_split = \
                self._best_split_finder.find(self._records, self._labels, possible_splits,
                                             self.height, self._feature_to_possible_splits[feature])
            if binary_decision_node_split is not None and (
                    not found_split or best_binary_decision_node_split.compare(
                    binary_decision_node_split) == -1):
                best_binary_decision_node_split = binary_decision_node_split
                found_split = True
        if found_split:
            return self._binary_decision_node_factory.create_with_node_split(
                self._records,
                self._labels,
                self.height,
                self.parent,
                self.is_left_child,
                best_binary_decision_node_split)
        return None

    def traverse_next_node(self, record):
        if self.split_feature is None:
            raise Exception("A node that is a leaf cannot be further traversed!")
        if record[self.split_feature] < self._less_than_split_point:
            return self.left_child
        else:
            return self.right_child

    def predict_label(self):
        # TODO consider ties
        if not self.is_leaf():
            raise Exception("A node that isn't a leaf cannot predict a label!")
        return self._labels.mode()[0]


class BinaryDecisionNodeFactory:

    def __init__(self, possible_split_finder, best_split_finder):
        self._possible_split_finder = possible_split_finder
        self._best_split_finder = best_split_finder

    def create_with_node_split(self, records, labels, height, parent, is_left_child, node_split):
        return BinaryDecisionNode.with_split(records, labels, height, parent, is_left_child,
                                             self._possible_split_finder, self._best_split_finder,
                                             self, node_split)

    def create(self, records, labels, height, parent, is_left_child):
        return BinaryDecisionNode(records, labels, height, parent, is_left_child,
                                  self._possible_split_finder, self._best_split_finder, self)
