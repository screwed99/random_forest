import pandas as pd


class PossibleSplitFinder:

    def __init__(self):
        pass

    def find(self, records, labels, feature):
        feature_and_label = pd.concat([records[feature], labels], axis=1)
        sorted_by_feature = feature_and_label.sort_values([feature], ascending=[True])
        possible_splits = []
        last_label = None
        last_seen_val_of_last_label = None
        for index, row in sorted_by_feature.iterrows():
            curr_label = row['label']
            curr_val = row[feature]
            if last_label is None:
                last_label = curr_label
            elif last_label != curr_label and last_seen_val_of_last_label != curr_val:
                mid_point_val = (last_seen_val_of_last_label + curr_val) / 2.
                possible_splits.append(mid_point_val)
                last_label = curr_label
            last_seen_val_of_last_label = curr_val
        return possible_splits
