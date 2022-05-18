import os

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import acf

from machine_learning.time_series.eda import compute_autocorrelation


current_dirpath = os.path.dirname(__file__)


def test_compute_autocorrelation():
    """
    tests compute_autocorrelation() against statsmodels acf()
    """
    data_dirpath = "../../../notebooks/data/time_series"
    data_filename = "widget_sales.csv"
    data_filepath = os.path.join(current_dirpath, data_dirpath, data_filename)
    df = pd.read_csv(data_filepath)
    values = df.widget_sales.values
    n_lags = 30
    correlations = [compute_autocorrelation(values, i) for i in range(n_lags + 1)]
    np.testing.assert_almost_equal(correlations, acf(values, nlags=n_lags))
