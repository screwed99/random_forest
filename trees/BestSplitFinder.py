import pandas as pd
import math


class BestSplitFinder:

    def __init__(self, binary_decision_node_split_factory):
        self._binary_decision_node_split_factory = binary_decision_node_split_factory

    def find(self, records, labels, node_splits, height, feature, parent):
        parent_label_counts = self._get_label_counts(labels)
        parent_entropy = self._entropy(*parent_label_counts)
        feature_and_label = pd.concat([records[feature], labels], axis=1)
        sorted_by_feature = feature_and_label.sort_values([feature], ascending=[True])
        curr_index = 0
        best_seen = (0, None, None)
        for node_split in node_splits:
            curr_val = feature_and_label[feature][curr_index]
            while curr_val < node_split:
                curr_index += 1
                curr_val = feature_and_label[feature][curr_index]
            information_gain_of_split =\
                self._get_information_gain(sorted_by_feature['label'], curr_index, parent_entropy)
            if information_gain_of_split > best_seen[0]:
                best_seen = (information_gain_of_split, node_split, curr_index)
        if best_seen[0] == 0:
            return None
        return self._create_node_split(*best_seen, sorted_by_feature, records, labels, height,
                                       parent)

    def _entropy(self, n, m):
        p_n, p_m = n / float(n + m), m / float(n + m)
        return - p_n * math.log(p_n, 2) - p_m * math.log(p_m, 2)

    def _get_label_counts(self, labels):
        counts_by_label = labels.value_counts()
        if len(counts_by_label) == 1:
            return counts_by_label[0], 0
        return counts_by_label[0], counts_by_label[1]

    def _get_information_gain(self, sorted_labels, split_index, parent_entropy):
        left_label_counts = self._get_label_counts(sorted_labels[:split_index])
        left_entropy = self._entropy(*left_label_counts)
        left_weighting = len(sorted_labels[:split_index]) / len(sorted_labels)
        right_label_counts = self._get_label_counts(sorted_labels[split_index:])
        right_entropy = self._entropy(*right_label_counts)
        right_weighting = len(sorted_labels[split_index:]) / len(sorted_labels)
        entropy_after_split = left_weighting * left_entropy + right_weighting * right_entropy
        return parent_entropy - entropy_after_split

    def _create_node_split(self, split_metric, split_point, split_index, sorted_by_feature, records,
                           labels, height, parent):
        left_records = records.iloc[sorted_by_feature[:split_index].index]
        left_labels = labels.iloc[sorted_by_feature[:split_index].index]
        right_records = records.iloc[sorted_by_feature[split_index:].index]
        right_labels = labels.iloc[sorted_by_feature[split_index:].index]
        return self._binary_decision_node_split_factory.create(
            split_metric, split_point, left_records, left_labels, right_records, right_labels,
            height, parent)
