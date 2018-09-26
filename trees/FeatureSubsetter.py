import math


class FeatureSubsetter:

    def __init__(self, features):
        self._features = set(features)
        self._num_features = len(features)

    def subset(self, used_features, random_state):
        num_available_features = self._num_features - len(used_features)
        num_features_to_return = min(num_available_features, int(math.sqrt(self._num_features)))
        available_features = list(self._features - set(used_features))
        subset = []
        for x in range(num_features_to_return):
            i = random_state.rand_int(len(available_features))
            subset.append(available_features[i])
            available_features.pop(i)
        return subset
