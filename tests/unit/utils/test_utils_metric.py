import numpy as np
from sklearn.metrics import mean_squared_error

from machine_learning.utils.utils_metric import rmse


def test_rmse():
    """
    tests rmse()
    """
    n_obs = 100
    np.random.seed(0)
    signal_0 = np.random.rand(n_obs)
    signal_1 = np.random.rand(n_obs)
    assert rmse(signal_0, signal_1) == mean_squared_error(
        signal_0, signal_1, squared=False
    )
