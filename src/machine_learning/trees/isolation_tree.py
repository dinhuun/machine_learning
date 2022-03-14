from __future__ import annotations

import math
import random
from typing import Optional, Union
from warnings import warn

import numpy as np
from numpy import ndarray

EULER_GAMMA = 0.5772156649  # theoretical constant


class Node:
    """
    a node in IsolationTree
    """

    def __init__(
        self,
        X: ndarray,
        feature: Optional[int],
        feature_threshold: Optional[float],
        depth: int,
        left: Optional[Node],
        right: Optional[Node],
        node_type: str = "",
    ):
        """
        initializes IsolationTree node
        :param X: dataset
        :param feature: node feature
        :param feature_threshold: node feature threshold
        :param depth: node depth
        :param left: what is to its left
        :param right: what is to its right
        :param node_type: note type
        """
        self.n_samples = X.shape[0]
        self.feature = feature
        self.feature_threshold = feature_threshold
        self.depth = depth
        self.left = left
        self.right = right
        self.node_type = node_type


def average_path_length(n_samples: int) -> float:
    """
    computes average path length in an IsolationTree
    :param n_samples: number of samples
    :return: average path length
    """
    return 2 * (np.log(n_samples - 1) + EULER_GAMMA) - 2 * (n_samples - 1) / n_samples


class IsolationTree:
    """
    this tree partitions samples in such a way that
    if
     - anomalous samples have feature values different to normal samples
     - anomalous samples are few compared to normal samples
    then
     - they are more quickly isolated and have shorter paths to root. This path length is how they are detected.
    """

    def __init__(
        self, X: ndarray, max_depth: int, max_features: Union[float, int, str]
    ):
        """
        initializes IsolationTree to dataset
        :param X: data
        :param max_depth: desired max depth to grow tree
        :param max_features: desired max number of features used to grow tree
        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.depth = 0  # updated as tree grows, unused by tree, could be used by caller
        self.max_depth = max_depth
        self.max_features = max_features
        if isinstance(max_features, str):
            if max_features == "auto":
                self.max_features_ = int(math.sqrt(self.n_features))
            else:
                raise ValueError(
                    f'max_features {self.max_features} is not supported. Use "auto", int or float'
                )
        elif isinstance(max_features, int):
            if max_features > self.n_features:
                warn(
                    f"max_features {max_features} is greater than number of features {self.n_features} and will be set to that"
                )
            self.max_features_ = min(self.n_features, max_features)
        else:
            if not (0 < max_features <= 1):
                raise ValueError(
                    f"max_features must be in (0, 1], not {self.max_features}"
                )
            self.max_features_ = int(max_features * self.n_features)

        self.features = sorted(
            random.sample(range(self.n_features), self.max_features_)
        )
        self.feature: Optional[int] = None
        self.feature_threshold: Optional[float] = None
        self.n_external_nodes = 0
        self.average_length = average_path_length(self.n_samples)
        self.root = self.grow(X, 0)

    def grow(self, X: ndarray, depth: int) -> Node:
        """
        grows tree from dataset
        :param X: dataset
        :param depth: depth
        :return: tree of depth min(given depth, max_depth)
        """
        self.depth = depth
        if X.shape[0] <= 1 or depth >= self.max_depth:
            self.n_external_nodes += 1
            return Node(X, None, None, depth, None, None, "external")
        else:
            self.feature = random.choice(self.features)
            feature_values = X[:, self.feature]
            feature_min = min(feature_values)
            feature_max = max(feature_values)
            self.feature_threshold = random.uniform(feature_min, feature_max)
            mask = X[:, self.feature] < self.feature_threshold
            return Node(
                X,
                self.feature,
                self.feature_threshold,
                depth,
                left=self.grow(X[mask], depth + 1),
                right=self.grow(X[~mask], depth + 1),
                node_type="internal",
            )

    def path_length(self, x: ndarray, node: Node) -> float:
        """
        computes length of path from x to given node
        :param x: sample
        :param node: given node
        :return: path length
        """
        if node.node_type == "external":
            if node.n_samples <= 1:
                return float(node.depth)
            else:
                return node.depth + average_path_length(node.n_samples)
        else:
            feature = node.feature
            if x[feature] < node.feature_threshold:
                return self.path_length(x, node.left)  # type: ignore
            else:
                return self.path_length(x, node.right)  # type: ignore

    def decision_function(self, x: ndarray) -> float:
        """
        computes "decision" for x, usually to decide whether x is anomalous
        the larger this number is compared to those of other samples, the more anomalous it is
        :param x: sample
        :return: 2 ** (-path_length(x, root) / average_path_length)
        """
        path_length = self.path_length(x, self.root)
        decision = np.power(2, -path_length / self.average_length)
        return decision
