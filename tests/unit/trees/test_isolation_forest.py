import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from machine_learning.trees.isolation_forest import IsolationForest

from .soln_isolation_tree import (
    X_test_anomalous,
    X_test_normal,
    X_train_normal,
    Y_test_anomalous,
    Y_test_normal,
    Y_train_normal,
)


def test_IsolationForest():
    """
    tests IsolationForest
    """
    # that it does not initialize if contamination is out of range
    with pytest.raises(ValueError):
        IsolationForest(contamination=-0.1)
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.5)

    # that it initializes otherwise
    forest = IsolationForest(seed=0)

    # that it can fit to dataset
    forest.fit(X_train_normal)

    # that it can compute decisions
    decisions = forest.decision_function(X_train_normal)
    assert isinstance(decisions, np.ndarray)

    # that it can predict
    Y_train_normal_pred = forest.predict(X_train_normal)
    assert accuracy_score(Y_train_normal, Y_train_normal_pred) >= 0.9
    Y_test_normal_pred = forest.predict(X_test_normal)
    assert accuracy_score(Y_test_normal, Y_test_normal_pred) >= 0.9
    Y_test_anomalous_pred = forest.predict(X_test_anomalous)
    assert accuracy_score(Y_test_anomalous, Y_test_anomalous_pred) >= 0.9
