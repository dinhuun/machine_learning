import random as rn
from typing import List, Optional, Union
from warnings import warn

import numpy as np
from numpy import ndarray

from machine_learning.trees.isolation_tree import IsolationTree


class IsolationForest:
    """
    isolation forest of IsolationTrees
    F. T. Liu, K. M. Ting, Z. H. Zhou, Isolation Forest
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_samples: Union[float, int, str] = "auto",
        max_features: Union[int, str] = "auto",
        contamination: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        initializes IsolationForest to dataset
        :param n_trees: number of trees in forest
        :param max_samples: desired max number of samples to grow each tree in forest
        :param max_features: desired max number of features to grow each tree in forest
        :param contamination: amount of contamination, i.e. proportional of anomalous samples believed in dataset,
                              used to compute internal threshold which is used to predict whether sample is normal.
                              should be in [0.0, 0.5)
        :param seed: random seed
        """
        if contamination < 0.0 or contamination >= 0.5:
            raise ValueError(f"contamination {contamination} should be in [0.0, 0.5)")

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.seed = seed
        self.trees: List[IsolationTree] = []
        self.max_samples_: int = 0
        self.threshold_: float = 0.0

    def fit(self, X: ndarray):
        """
        fits forest to dataset
        :param X: dataset
        """
        n_samples = len(X)
        rn.seed(self.seed)

        if isinstance(self.max_samples, str):
            if self.max_samples == "auto":
                self.max_samples_ = min(
                    131072, n_samples
                )  # tree size 2^17, authors say 2^8 is large enough
            else:
                raise ValueError(
                    f'max_samples {self.max_samples} is not supported. Use "auto", int or float'
                )
        elif isinstance(self.max_samples, int):
            if self.max_samples > n_samples:
                warn(
                    f"max_samples {self.max_samples} is greater than number of samples {n_samples} and will be set to that"
                )
            self.max_samples_ = min(self.max_samples, n_samples)
        else:
            if not (0 < self.max_samples <= 1):
                raise ValueError(
                    f"max_samples must be in (0, 1], not {self.max_samples}"
                )
            self.max_samples_ = int(self.max_samples * n_samples)

        if self.max_samples_ == 1:
            raise ValueError("there is only one sample")

        max_depth = int(np.ceil(np.log2(self.max_samples_)))

        for i in range(self.n_trees):
            X_sampled = X[rn.sample(range(n_samples), self.max_samples_)]
            self.trees.append(IsolationTree(X_sampled, max_depth, self.max_features))

        self.threshold_ = -np.percentile(
            -self.decision_function(X), 100 * (1 - self.contamination)
        )

    def decision_function(self, X: ndarray) -> ndarray:
        """
        computes "decision" for x in X by averaging decisions by trees in forest. See IsolationTree.decision_function().
        :param X: samples
        :return: one decision for each x in X
        """
        if X.ndim == 1:
            tree_decisions = [tree.decision_function(X) for tree in self.trees]
            return np.array(np.mean(tree_decisions))
        decisions = np.zeros(len(X))
        for i in range(len(X)):
            decisions[i] = self.decision_function(X[i])
        return decisions

    def predict(self, X: ndarray) -> ndarray:
        """
        predicts whether x in X is normal or anomalous
        :param X: samples
        :return: 1 for normal x, 0 for anomalous x
        """
        is_normal = np.ones(len(X), dtype=int)
        is_normal[1 - self.decision_function(X) < self.threshold_] = 0
        return is_normal
