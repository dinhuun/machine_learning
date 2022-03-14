import numpy as np
from numpy import ndarray


def mape(X_0: ndarray, X_1: ndarray) -> float:
    """
    computes mean absolute percentage error
    :return: between 0 and 1, to be multiplied by 100 for xy%
    """
    e = X_0 - X_1
    pe = e / X_0
    ape = np.abs(pe)
    return np.mean(ape)


def rmse(X_0: ndarray, X_1: ndarray) -> float:
    """
    computes root mean squared error between 2 arrays of equal length
    agrees with sklearn.metrics.mean_squared_error(X_0, X_1, squared=False)
    """
    len_X_0 = len(X_0)
    len_X_1 = len(X_1)
    if len_X_0 != len_X_1:
        raise ValueError("arrays have different lengths")
    e = X_0 - X_1
    se = np.power(e, 2)
    mse = np.mean(se)
    return np.sqrt(mse)
