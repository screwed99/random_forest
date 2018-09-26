from trees.NodeSplitResult import NodeSplitResult


class NodeSplitter:

    def split(self, leaf_nodes, allowed_features):
        best_node_after_split = None
        best_node_before_split = None
        found_split = False
        for leaf_node in leaf_nodes:
            node_after_split = leaf_node.get_best_split(allowed_features)
            if node_after_split is not None:
                if not found_split or node_after_split.best_binary_decision_node_split.compare(
                        best_node_after_split.binary_decision_node_split) == -1:
                    best_node_after_split = node_after_split
                    best_node_before_split = leaf_node
                    found_split = True
        return NodeSplitResult(found_split=found_split,
                               best_node_before_split=best_node_before_split,
                               best_node_after_split=best_node_after_split)
