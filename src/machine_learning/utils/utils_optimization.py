from typing import Dict

import numpy as np
from numpy import ndarray


def report_hyperparameter_search_result(results: Dict[str, ndarray], n_top: int = 3):
    """
    reports sklearn GridSearchCV.cv_results and RandomizedSearchCV.cv_results
    :param results: GridSearchCV.cv_results or RandomizedSearchCV.cv_results
    :param n_top: number of top results
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            mean_validation_score = results["mean_test_score"][candidate]
            std_validation_score = results["std_test_score"][candidate]
            params = results["params"][candidate]
            print(f"model with rank: {i}")
            print(f"mean validation score: {mean_validation_score:.3f}")
            print(f"std validation score: {std_validation_score: .3f}")
            print(f"parameters: {params}")
            print("\n")
