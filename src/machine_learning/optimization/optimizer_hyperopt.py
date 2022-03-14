from functools import partial
from typing import Dict

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from numpy.polynomial.polynomial import Polynomial


def objective(space, a: float, b: float, c: float) -> Dict[str, float]:
    """
    this uni-valued function computes a*x**2 + b*x + c for given x, to be optimized by hyperopt
    :param space: space of values for x
    :param a: 2nd-degree coefficient
    :param b: 1st-degree coefficient
    :param c: constant coefficient
    :return: dictionary {"loss": loss value}
    """
    loss = Polynomial((c, b, a))
    x = space["x"]
    return {"loss": loss(x), "status": STATUS_OK}


def optimize_objective(
    a: float,
    b: float,
    c: float,
    lower_bound: float,
    upper_bound: float,
    n_trials: int = 100,
):
    """
    optimizes objective()
    this is an example of uni-objective optimization by optimization package hyperopt
    """
    _objective = partial(objective, a=a, b=b, c=c)
    space = {"x": hp.uniform("x", lower_bound, upper_bound)}
    trials = Trials()
    best_params = fmin(
        _objective, space, tpe.suggest, n_trials, trials=trials, verbose=True
    )
    return best_params
