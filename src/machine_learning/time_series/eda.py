import numpy as np
from numpy import ndarray


def compute_autocorrelation(s: ndarray, lag: int = 2) -> float:
    """
    computes correlation between a series and its lagged version
    :param s: series
    :param lag: time lag
    :return: correlation between a series and its lagged version
    """
    if lag == 0:
        return 1.0
    s_bar = np.mean(s)
    num_0 = s[lag:] - s_bar
    num_1 = s[:-lag] - s_bar
    num = sum(num_0 * num_1)
    dem = sum((s - s_bar) ** 2)
    return num / dem
