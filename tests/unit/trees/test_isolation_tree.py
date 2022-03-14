import pytest

from machine_learning.trees.isolation_tree import IsolationTree, Node

from .soln_isolation_tree import X_train_normal, max_depth, mu_normal


def test_IsolationTree():
    """
    tests IsolationTree
    """
    # that it does not initialize if str max_features is not "auto"
    with pytest.raises(ValueError):
        IsolationTree(X_train_normal, max_depth, "not_auto")

    # that it does not initialize if float max_features is out of (0.0, 1.0]
    with pytest.raises(ValueError):
        IsolationTree(X_train_normal, max_depth, 0.0)
    with pytest.raises(ValueError):
        IsolationTree(X_train_normal, max_depth, 1.1)

    # that it initializes otherwise
    tree = IsolationTree(X_train_normal, max_depth, "auto")

    # that it can grow
    assert isinstance(tree.grow(X_train_normal, 0), Node)

    # that it can compute path length
    path_length = tree.path_length(mu_normal, tree.root)
    assert isinstance(path_length, float)

    # that it can compute decision
    decision = tree.decision_function(mu_normal)
    assert 0.0 <= decision <= 1.0
