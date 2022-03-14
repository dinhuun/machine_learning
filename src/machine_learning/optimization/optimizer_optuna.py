from functools import partial
from typing import Tuple

import optuna as opt
from numpy.polynomial.polynomial import Polynomial
from optuna.trial import Trial


def uni_objective(
    trial: Trial, a: float, b: float, c: float, lower_bound: float, upper_bound: float
) -> Tuple[float, float]:
    """
    this uni-valued function computes a*x**2 + b*x + c for given x, to be optimized by optuna
    :param trial: an optuna Trial
    :param a: 2nd-degree coefficient
    :param b: 1st-degree coefficient
    :param c: constant coefficient
    :param lower_bound: lower bound for x
    :param upper_bound: upper bound for x
    :return: a*x**2 + b*x + c
    """
    f = Polynomial((c, b, a))
    x = trial.suggest_uniform("x", lower_bound, upper_bound)
    return f(x)


def multi_objective(trial: Trial) -> Tuple[float, float]:
    """
    this multi-valued function computes (a + b, a - b) for given (a, b), to be optimized by optuna for Pareto optimality
    :param trial: an optuna Trial
    :return: (a + b, a - b)
    """
    a = trial.suggest_uniform("a", 0, 1)
    b = trial.suggest_uniform(
        "b", 0, 1 - a
    )  # dynamically changed upper bound, which TPESampler sometimes violates
    if b > 1 - a:
        raise ValueError(f"optuna sampled b {b} out of bounds {0, 1 - a}")
    sum_0 = a + b
    sum_1 = a - b
    return sum_0, sum_1


def optimize_uni_objective(
    a: float,
    b: float,
    c: float,
    lower_bound: float,
    upper_bound: float,
    n_trials: int = 100,
) -> opt.study.Study:
    """
    optimizes uni_objective()
    this is an example of uni-objective optimization by optimization package optuna
    """
    study = opt.create_study()
    _uni_objective = partial(
        uni_objective, a=a, b=b, c=c, lower_bound=lower_bound, upper_bound=upper_bound
    )
    study.optimize(_uni_objective, n_trials=n_trials)
    return study


def optimize_multi_objective(n_trial: int = 100):
    """
    optimizes multi_objective()
    this is
     - an example of multi-objective optimization by optimization package optuna for Pareto optimality
     - and example of how its default sampler TPESampler sometimes violates dynamically changed bounds.
       it seems this sampler will start to sample based on previous values at around 50th trial, sometimes violating
       dynamically changed bounds. See unit test.
    """
    study = opt.create_study(directions=["maximize", "maximize"])
    study.optimize(multi_objective, n_trials=n_trial)
    best_trials = study.best_trials
    print(best_trials)
