class LeafNodes:

    def __init__(self, leaf_nodes):
        self._leaf_nodes = set(leaf_nodes)

    def get_leaf_nodes(self):
        return list(self._leaf_nodes)

    def update_leaf_node_split(self, best_node_before_split, best_node_after_split):
        self._leaf_nodes.remove(best_node_before_split)
        self._leaf_nodes.add(best_node_after_split.left_child)
        self._leaf_nodes.add(best_node_after_split.right_child)
