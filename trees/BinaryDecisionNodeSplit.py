class BinaryDecisionNodeSplit:

    def __init__(self, split_metric, split_point, child_left, child_right):
        self.split_metric = split_metric
        self.split_point = split_point
        self.child_left = child_left
        self.child_right = child_right

    def compare(self, other_binary_decision_node_split):
        if self.split_metric == other_binary_decision_node_split.split_metric:
            return 0
        elif self.split_metric < other_binary_decision_node_split.split_metric:
            return -1
        else:
            return 1


class BinaryDecisionNodeSplitFactory:

    def __init__(self, binary_decision_node_factory):
        self._binary_decision_node_factory = binary_decision_node_factory

    def create(self, split_metric, split_point, left_records, left_labels, right_records,
               right_labels, parent_height, parent):
        child_left = self._binary_decision_node_factory.create(
            left_records, left_labels, parent_height + 1, parent, True)
        child_right = self._binary_decision_node_factory.create(
            right_records, right_labels, parent_height + 1, parent, False)
        return BinaryDecisionNodeSplit(split_metric, split_point, child_left, child_right)
