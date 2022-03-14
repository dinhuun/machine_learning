import numpy as np
import pytest

from machine_learning.optimization.optimizer_optuna import (
    optimize_multi_objective,
    optimize_uni_objective,
)


def test_optimize_uni_objective():
    """
    tests optimize_uni_objective()
    """
    a = 1
    b = -2
    c = 1
    lower_bound = 0
    upper_bound = 2
    n_trials = 100
    study = optimize_uni_objective(a, b, c, lower_bound, upper_bound, n_trials)
    np.testing.assert_almost_equal(study.best_params["x"], 1, decimal=2)


def test_optimize_multi_objective():
    """
    tests optimize_multi_objective()
    """
    # that default sampler TPESampler sometimes violates dynamically changed bounds.
    # it seems this sampler will start to sample based on previous values at around 50th trial,
    # sometimes violating dynamically changed bounds
    with pytest.raises(ValueError):
        optimize_multi_objective(100)
