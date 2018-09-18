import math


class FeatureSubsetter:

    def __init__(self):
        pass

    def subset(self, features, used_features, random_state):
        num_features = len(features)
        num_available_features = num_features - len(used_features)
        num_features_to_return = min(num_available_features, int(math.sqrt(num_features)))
        available_features = list(set(features) - set(used_features))
        subset = []
        for x in range(num_features_to_return):
            i = random_state.rand_int(len(available_features))
            subset.append(available_features[i])
            available_features.pop(i)
        return subset
