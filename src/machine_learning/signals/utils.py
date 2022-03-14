from typing import Tuple

import numpy as np
from numpy import ndarray


def stack_windows(
    obs: ndarray, window_size: int = 1, vertical: bool = False
) -> Tuple[ndarray, ndarray]:
    """
    stacks sliding window_size windows of obs
    example: obs [0, 1, 2, 3, 4] and window_size 2 produce [[0, 1], [1, 2], [2, 3]] and [2, 3, 4]
    :param obs: observations
    :param window_size: window size
    :param vertical: if true then each window is (window_size, 1), else each window is (1, window_size)
    :return: windows and their subsequent obs
    """
    windows, subsequent_obs = [], []
    for i in range(len(obs) - window_size):
        x = obs[i : i + window_size]
        windows.append(x)
        y = obs[i + window_size]
        subsequent_obs.append(y)
    X = np.array(windows)
    Y = np.array(subsequent_obs)
    if vertical is True:
        X = X.reshape(-1, window_size, 1)
    else:
        X = X.reshape(-1, 1, window_size)
    return X, Y
