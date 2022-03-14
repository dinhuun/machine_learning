import numpy as np
from numpy import ndarray


def hard_arg_max(X: ndarray) -> ndarray:
    """
    one-hot encodes arg max of array
    :param X: 1-dim array
    :return: array of 0 and 1, where 1 appears at max index
    """
    max_index = np.argmax(X)
    y = np.zeros(len(X), dtype=np.int16)
    y[max_index] = 1
    return y


def soft_arg_max(X: ndarray, t: float = 1) -> ndarray:
    """
    smooths arg max of array
    :param X: 1-dim array
    :param t: known as temperature
              as t goes to 0, soft_arg_max goes to hard_arg_max
              as t goes to inf, soft_arg_max goes to uniform [1/n,..., 1/n]
    :return: array of floats between 0 and 1
    """
    exponentiated = np.exp(X / t)
    y = exponentiated / np.sum(exponentiated)
    return y


def link_relatives(array: ndarray) -> ndarray:
    """
    divides array[1:] by previous entries
    """
    return array[1:] / array[:-1]


def raise_not_implemented_error(obj_str: str, obj_name: str):
    """
    raises NotImplementedError when named object has not been implemented
    :param obj_str: object, such as regularizer
    :param obj_name: object name, such as "l3"
    """
    raise NotImplementedError(f"{obj_str} {obj_name} has not been implemented")
