import numpy as np

from machine_learning.optimization.optimizer_hyperopt import optimize_objective


def test_optimize_objective():
    """
    tests optimize_objective()
    """
    a = 1
    b = -2
    c = 1
    lower_bound = 0
    upper_bound = 2
    n_trials = 100
    best_params = optimize_objective(a, b, c, lower_bound, upper_bound, n_trials)
    np.testing.assert_almost_equal(best_params["x"], 1, decimal=2)
